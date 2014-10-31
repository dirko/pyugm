"""
Module containing classes to learn parameters from examples.
"""
# License: BSD 3 clause

import numpy as np
import scipy.optimize

from pyugm.infer import LoopyBeliefUpdateInference
from pyugm.infer import FloodingProtocol


class LearnMRFParameters(object):
    """
    Find a Gaussian approximation to the posterior given a model and prior.
    """
    def __init__(self, model, prior=1.0, initial_noise=10.0**-12, update_order=None):
        """
        The learner object.

        :param prior: Float representing the prior sigma squared of all parameters.
        """
        self._model = model

        # Get number of parameters and map parameters to position in parameter vector
        parameter_set = set()
        for factor in self._model.factors:
            if factor.parameters is not None:
                for parameter in factor.parameters.reshape(-1, ):
                    if isinstance(parameter, str):
                        parameter_set.add(parameter)
        self._index_to_parameters = {}
        self._parameters_to_index = {}
        for index, key in enumerate(sorted(list(parameter_set))):
            self._index_to_parameters[index] = key
            self._parameters_to_index[key] = index

        self._dimension = len(self._index_to_parameters)
        self._parameters = np.random.randn(self._dimension) * initial_noise
        if self._dimension > 0:
            self._prior_location = np.zeros(self._dimension)
            self._prior_precision = np.eye(self._dimension) * prior
            self._prior_normaliser = (-0.5 * self._dimension * np.log(2.0 * np.pi)
                                      + 0.5 * np.log(np.linalg.det((self._prior_precision))))
        self._update_order = update_order
        if not self._update_order:
            self._update_order = FloodingProtocol(self._model)

        # Results
        self._iterations = []
        self._optimizer_result = None

    @property
    def parameters(self):
        """
        Helper to update the potential tables given the parameters.
        """
        parameter_dictionary = {}
        for i, value in enumerate(self._parameters):
            parameter_dictionary[self._index_to_parameters[i]] = value
        return parameter_dictionary

    def evaluate_log_likelihood_gradient(self, evidence):
        """
        Run inference on the model to find the log-likelihood of the model given evidence and its gradient with respect
            to the model parameters.
        :param evidence: A dictionary where the key is a variable name and the value its observed value.
        :returns: The log-likelihood and a vector of derivatives.
        """
        self._update_order.reset()
        self._model.set_parameters(self.parameters)
        inference = LoopyBeliefUpdateInference(self._model)
        inference.calibrate(update_order=self._update_order)
        log_z_total = self._model.factors[0].log_normalizer
        model_expected_counts = self._accumulate_expected_counts()

        self._update_order.reset()
        self._model.set_parameters(self.parameters)
        self._model.set_evidence(evidence=evidence)
        inference = LoopyBeliefUpdateInference(self._model)
        inference.calibrate(update_order=self._update_order)
        log_z_observed = self._model.factors[0].log_normalizer
        empirical_expected_counts = self._accumulate_expected_counts()

        log_likelihood = log_z_observed - log_z_total
        derivative = empirical_expected_counts - model_expected_counts

        if self._dimension > 0:
            derivative += -np.dot(self._prior_precision, (self._parameters - self._prior_location))
            log_likelihood += -0.5 * np.dot(np.dot((self._parameters - self._prior_location).T,
                                                   self._prior_precision), (self._parameters - self._prior_location))
            log_likelihood += self._prior_normaliser
        return log_likelihood, derivative

    def _accumulate_expected_counts(self):
        """
        Iterate through factors and add parameter values.
        :returns: Vector of expected counts for each parameter.
        """
        expected_counts = np.zeros(self._parameters.shape)
        for factor in self._model.factors:
            factor_sum = np.sum(factor._data)
            for parameter, value in zip(factor.parameters.flatten(), factor._data.flatten()):
                if isinstance(parameter, str):
                    expected_counts[self._parameters_to_index[parameter]] += (value / factor_sum)  # * norm / normalizer
        return expected_counts

    def fit_without_gradient(self, evidence, initial_parameters=None):
        """
        Infer Gaussian approximation to the posterior using gradient-less BFGS optimization.
        :param evidence: Dictionary where the key is a variable name and the value the observed value of that variable.
        :param initial_parameters: np.array of initial parameter values. If None then random values around 0 is used.
        :return: The learner object.
        """
        x0 = self._parameters
        if initial_parameters is not None:
            x0 = initial_parameters

        def f(x):
            """
            Function that is passed to the optimizer.
            :param x: Parameter vector.
            :returns: Negative log-likelihood.
            """
            self._parameters = x
            ll, _ = self.evaluate_log_likelihood_gradient(evidence)
            self._iterations.append([ll, x])
            return -ll

        self._optimizer_result = scipy.optimize.fmin_l_bfgs_b(f, x0, approx_grad=True, pgtol=10.0**-10)
        return self

    def fit(self,
            evidence,
            initial_parameters=None,
            optimizer=scipy.optimize.fmin_l_bfgs_b,
            optimizer_kwargs={'pgtol': 10.0**-10}):
        """
        Fit the model to the data.
        :param evidence: Dictionary where the key is a variable name and the value the observed value of that variable.
        :param initial_parameters: np.array of initial parameter values. If None then random values around 0 is used.
        :param optimizer: The optimization function to use.
        :param optimizer_kwargs: Keyword arguments that are passed to the optimizer.
        """
        x0 = self._parameters
        if initial_parameters is not None:
            x0 = initial_parameters

        def f(x):
            """
            Function that is passed to the optimizer.
            :param x: Parameter vector.
            :returns: Negative log-likelihood. gradient.
            """
            self._parameters = x
            ll, grad = self.evaluate_log_likelihood_gradient(evidence)
            self._iterations.append([ll, x])
            return -ll, -grad

        self._optimizer_result = optimizer(f, x0, **optimizer_kwargs)
        return self

    def result(self):
        """
        Get the result object after learning.
        :returns: Log-likelihood, parameters that maximises the log-likelihood.
        """
        if self._optimizer_result:
            return self._optimizer_result[1], self._optimizer_result[0]
        else:
            raise Exception('No result yet - run fit.')

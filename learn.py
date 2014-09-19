from factor import DiscreteFactor
from infer import Model, LoopyBeliefUpdateInference
import numpy as np
import scipy.optimize


class LearnMRFParameters:
    """
    Find a Gaussian approximation to the posterior given a model and prior.
    """
    def __init__(self, model, prior=1.0, initial_noise=0.1):
        """
        The learner object.

        :param prior: Float representing the prior sigma squared of all parameters.
        """
        self.model = model
        self.dimension = len(self.model.parameters_to_index)
        self.parameters = np.random.randn(self.dimension) * initial_noise
        if self.dimension > 0:
            self.prior_location = np.zeros(self.dimension)
            self.prior_precision = np.eye(self.dimension) * prior
            self.prior_normaliser = (-0.5 * self.dimension * np.log(2.0 * np.pi)
                                     + 0.5 * np.log(np.linalg.det((self.prior_precision))))

    def evaluate_log_likelihood(self, evidence):
        self.model.set_parameters(self.parameters)
        inference = LoopyBeliefUpdateInference(self.model)
        inference.set_up_belief_update()
        change = inference.update_beliefs(number_of_updates=35)
        log_Z_total = np.log(self.model.factors[0].data.sum()) + self.model.factors[0].log_normalizer

        self.model.set_parameters(self.parameters)
        self.model.set_evidence(evidence=evidence)
        inference = LoopyBeliefUpdateInference(self.model)
        inference.set_up_belief_update()
        change = inference.update_beliefs(number_of_updates=35)
        log_Z_observed = np.log(self.model.factors[0].data.sum()) + self.model.factors[0].log_normalizer

        log_likelihood = log_Z_observed - log_Z_total

        if self.dimension > 0:
            log_likelihood += -0.5 * np.dot(np.dot((self.parameters - self.prior_location).T,
                                                   self.prior_precision), (self.parameters - self.prior_location))
            log_likelihood += self.prior_normaliser
        print 'Z: ', log_likelihood, '   ', log_Z_observed, log_Z_total
        return log_likelihood

    def evaluate_log_likelihood_and_derivative(self, evidence):
        self.model.set_parameters(self.parameters)
        inference = LoopyBeliefUpdateInference(self.model)
        inference.set_up_belief_update()
        change = inference.update_beliefs(number_of_updates=35)
        log_Z_total = np.log(self.model.factors[0].data.sum()) + self.model.factors[0].log_normalizer
        model_expected_counts = self.accumulate_expected_counts()

        self.model.set_parameters(self.parameters)
        self.model.set_evidence(evidence=evidence)
        inference = LoopyBeliefUpdateInference(self.model)
        inference.set_up_belief_update()
        change = inference.update_beliefs(number_of_updates=35)
        log_Z_observed = np.log(self.model.factors[0].data.sum()) + self.model.factors[0].log_normalizer
        empirical_expected_counts = self.accumulate_expected_counts()

        log_likelihood = log_Z_observed - log_Z_total
        derivative = model_expected_counts - empirical_expected_counts

        if self.dimension > 0:
            derivative += np.dot(self.prior_precision, (self.parameters - self.prior_location))
            log_likelihood += -0.5 * np.dot(np.dot((self.parameters - self.prior_location).T,
                                                   self.prior_precision), (self.parameters - self.prior_location))
            log_likelihood += self.prior_normaliser
        print 'Z: ', log_likelihood, '   ', log_Z_observed, log_Z_total
        return log_likelihood, derivative

    def accumulate_expected_counts(self):
        """
        Go through factors and add parameter values.
        """
        expected_counts = np.zeros(self.parameters.shape)
        for factor in self.model.factors:
            factor_sum = np.sum(factor.data)
            for parameter, value in zip(factor.parameters.flatten(), factor.data.flatten()):
                expected_counts[self.model.parameters_to_index[parameter]] += (value / factor_sum)  # * norm / normalizer
        return expected_counts

    def fit_without_gradient(self, evidence, initial_parameters=None):
        """
        Infer Gaussian approximation to the posterior using gradient-less BFGS optimization.
        :param evidence: Dict containing {variable_1: value_1, ...} for different variables.
        :initial_parameters: np.array of initial parameter values. If None then random values arount 0 with is used.
        :return: The learner object.
        """
        self.evidence = evidence
        x0 = self.parameters
        if initial_parameters is not None:
            x0 = initial_parameters

        def f(x):
            self.parameters = x
            print np.exp(x)
            return -self.evaluate_log_likelihood(self.evidence)

        self.ans = scipy.optimize.fmin_l_bfgs_b(f, x0, approx_grad=True, pgtol=10.0**-10)
        return self

    def fit(self, evidence, initial_parameters=None):
        self.evidence = evidence
        x0 = self.parameters
        if initial_parameters is not None:
            x0 = initial_parameters

        def f(x):
            self.parameters = x
            print np.exp(x)
            ll, grad = self.evaluate_log_likelihood_and_derivative(self.evidence)
            return -ll, grad

        self.ans = scipy.optimize.fmin_l_bfgs_b(f, x0, pgtol=10.0**-10)
        return self

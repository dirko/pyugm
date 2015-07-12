"""
Module containing the inference class.
"""
# License: BSD 3 clause

import numpy
import abc
from pyugm.factor import DiscreteBelief


class Inference(object):
    """
    An inference object to calibrate the potentials.
    """
    def __init__(self, model):
        """
        Constructor.
        :param model: The model.
        """
        self._model = model
        self.beliefs = {factor: DiscreteBelief(factor) for factor in self._model.factors}
        self.parameters = None
        self._separator_potential = dict()  # edge to pairs of separator potentials
        self._set_up_separators()
        self._original_beliefs = None  # The beliefs just before inference started

    def _set_up_separators(self):
        """
        Helper to initialise separator potentials to 1.
        """
        for edge in list(self._model.edges):
            factor1, factor2 = edge
            sepset = factor1.variable_set.intersection(factor2.variable_set)
            separator_variables = [(variable, factor1.cardinalities[variable]) for variable in list(sepset)]
            # NOTE: will want to set this up more generically for other kinds of factors
            separator_belief = DiscreteBelief(variables=separator_variables)
            self._separator_potential[edge] = separator_belief
            self._separator_potential[(edge[1], edge[0])] = separator_belief

    def _set_evidence(self, evidence):
        """
        Set the evidence in each of the factors contained in the model.
        :param evidence: Dictionary where the key is a variable name and the value is the observed value of that
            variable.
        """
        for variable, value in evidence.items():
            for factor in self._model.variables_to_factors[variable]:
                belief = self.beliefs[factor]
                belief.set_evidence({variable: value})

    def get_marginals(self, variable):
        """
        Get marginals of all the factors in which a variable appears.
        :param variable: The variable.
        :returns: List of factors.
        """
        return [self.beliefs[factor].marginalize([variable]) for factor in self._model.variables_to_factors[variable]]

    def _set_parameters(self, parameters, noise_variance=0.0):
        """
        Fill factor potentials with exponential of the parameters.
        :param parameters: Dictionary where the key is a parameter name and the value the log value of the parameter.
        """
        self.parameters = parameters
        for belief in self.beliefs.values():
            belief.set_parameters(dict((key, value + numpy.random.randn() * noise_variance)
                                       for key, value in parameters.items()))

    def partition_approximation(self):
        """
        Calculate the factored energy functional approximation to the partition function by using the current beliefs.
        :returns: Approximation to Z.
        """
        energy = 0.0
        for factor, belief in self.beliefs.items():
            reduced_factor = self._original_beliefs[factor]
            energy += (numpy.log(reduced_factor.data + 10.0**-20) * belief.normalized_data).sum() + self._entropy(belief)
        for separator_belief in set(self._separator_potential.values()):
            energy -= self._entropy(separator_belief)
        return energy

    @staticmethod
    def _entropy(belief):
        """ Helper to calculate the entropy of a belief.  """
        return -(belief.normalized_data * numpy.log(belief.normalized_data + 10.0**-10)).sum()

    def calibrate(self, evidence=None, parameters=None):
        """
        Calibrate all the factors in the model by running belief updates according to the `update_order` ordering
        scheme.
        """
        if parameters:
            self._set_parameters(parameters)
        else:
            self._set_parameters({})
        if evidence:
            self._set_evidence(evidence)
        else:
            evidence = {}
        self._original_beliefs = {key: DiscreteBelief(belief) for key, belief in self.beliefs.items()}
        self._calibrate(evidence)
        return self

    @abc.abstractmethod
    def _calibrate(self, evidence):
        """
        Calibrate all the factors in the model by running belief updates according to the `update_order` ordering
        scheme. The public `calibrate` method wraps this abstract method.
        """

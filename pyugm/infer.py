"""
Module containing the inference class.
"""
# License: BSD 3 clause

import numpy
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

    def set_evidence(self, evidence):
        """
        Set the evidence in each of the factors contained in the model.
        :param evidence: Dictionary where the key is a variable name and the value is the observed value of that
            variable.
        """
        for variable, value in evidence.items():
            for factor in self._model.variables_to_factors[variable]:
                belief = self.beliefs[factor]
                belief.set_evidence({variable: value})

    def get_evidence(self):
        """
        Retrieves the evidence from each factor.
        :returns: A dictionary where the key is a variable name and the value is the observed value.
        """
        evidence = {}
        for factor, belief in self.beliefs.items():
            for variable, value in belief.evidence.items():
                evidence[variable] = value  # overwrites possibly conflicting values
        return evidence

    def get_marginals(self, variable):
        """
        Get marginals of all the factors in which a variable appears.
        :param variable: The variable.
        :returns: List of factors.
        """
        return [self.beliefs[factor].marginalize([variable]) for factor in self._model.variables_to_factors[variable]]

    def set_parameters(self, parameters, noise_variance=0.0):
        """
        Fill factor potentials with exponential of the parameters.
        :param parameters: Dictionary where the key is a parameter name and the value the log value of the parameter.
        """
        for belief in self.beliefs.values():
            belief.set_parameters(dict((key, value + numpy.random.randn() * noise_variance)
                                       for key, value in parameters.items()))

    def partition_approximation(self):
        """
        Calculate the factored energy functional approximation to the partition function by using the current beliefs.
        :returns: Approximation to Z.
        """
        # TODO
        return 1.0
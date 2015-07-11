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
        self.parameters = None
        self._separator_potential = dict()  # edge to pairs of separator potentials
        self._set_up_separators()

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
            reduced_factor = DiscreteBelief(factor)
            reduced_factor.set_parameters(self.parameters)
            reduced_factor.set_evidence(self.get_evidence())

            energy += (numpy.log(reduced_factor.data + 10.0**-20) * belief.normalized_data).sum() + self._entropy(belief)
            print factor, belief, energy, (numpy.log(factor.data) * belief.data).sum(),\
                (numpy.log(reduced_factor.data + 10.0**-20) * belief.data).sum(), self._entropy(belief)
        for separator_belief in set(self._separator_potential.values()):
            energy -= self._entropy(separator_belief)
            print factor, belief, energy
        return energy

    @staticmethod
    def _entropy(belief):
        """ Helper to calculate the entropy of a belief.  """
        return -(belief.normalized_data * numpy.log(belief.normalized_data + 10.0**-10)).sum()
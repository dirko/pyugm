"""
Module containing the inference routines.
"""
# License: BSD 3 clause

import numpy
from numba import jit
from pyugm.infer import Inference


class GibbsSamplingInference(Inference):
    """
    An inference object to calibrate the potentials.
    """
    def __init__(self, model):
        """
        Constructor.
        :param model: The model.
        """
        super(GibbsSamplingInference, self).__init__(model)
        self.traces = {}

    def calibrate(self, initial_values=None, samples=1000, burn_in=100, callback=None):
        """
        Calibrate all the factors in the model.
        """
        evidence = self.get_evidence()

        if initial_values:
            self.traces = {variable: [initial_value] for variable, initial_value in initial_values.items()}
        elif not self.traces:
            # TODO: Only valid initialisation should be allowed.
            self.traces = {variable: [numpy.random.randint(0, self._model.cardinalities[variable])]
                           for variable in self._model.variables if variable not in evidence}

        for sample in xrange(samples):
            if callback:
                callback(sample, self.traces)
            for variable, factors in self._model.variables_to_factors.items():
                if variable not in evidence:
                    beliefs = [self.beliefs[factor] for factor in factors]
                    self.traces[variable].append(self._sample(variable, beliefs, self.traces, evidence))

        # Reset the factor potentials
        for belief in self.beliefs.values():
            belief._data = numpy.zeros(belief._data.shape)

        # Add traces to factor tables
        for belief in self.beliefs.values():
            for i in xrange(burn_in, samples):
                assignment = tuple([self.traces[variable][i] if variable in self.traces else evidence[variable]
                                    for _, variable in belief.axis_to_variable.items()])
                belief._data.__setitem__(assignment, belief._data.__getitem__(assignment) + 1)
            belief.normalize()

    def _sample(self, current_variable, factors, traces, evidence):
        """
        Helper to sample a new value for a variable.
        :param current_variable: The current variable.
        :param factors: All the factors that contain the variable.
        :param traces: The traces of all the variables so far.
        """
        dist = numpy.ones(self._model.cardinalities[current_variable])
        for factor in factors:
            assignment_list = [(variable, evidence[variable] if variable in evidence else traces[variable][-1])
                               for variable, _ in factor.variables if current_variable != variable]
            dist *= factor.get_potential(assignment_list)
        return sample_categorical(dist)


@jit
def sample_categorical(probability_table):
    """
    Helper to sample from the categorical distribution.
    :param probability_table: Numpy array where the N'th entry corresponds to the probability of the N'th category.
    :returns: integer.
    """
    cumulative = probability_table.cumsum()
    random_number = numpy.random.ranf() * cumulative[-1]
    for i, val in enumerate(cumulative):
        if random_number < val:
            return i

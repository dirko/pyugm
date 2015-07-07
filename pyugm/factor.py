"""
Module containing the factor and belief classes.
"""
# License: BSD 3 clause

import numpy


class DiscreteFactor(object):
    """
    A factor containing only discrete variables. Factors are immutable and basically a container for a probability table
    and its metadata.
    """

    def __init__(self, variables, data=None, parameters=None):
        """
        Constructor.

        :param variables: A list of (variable_name, cardinality) tuples, where variable_name is a string or int.
            If only a list of names is provided, cardinalities are assumed to be 2.
        :param data: ndarray of factor potentials.
        :param parameters: ndarray of parameter names (strings or integers). Must be the same shape as the potential
            table.
        """
        # variables = [(name, cardinality), (name, cardinality) ... ]
        # parameters = array (dtype=object) of same shape as data, containing parameter names

        # Short form: if only list of names, use cardinalities of 2
        try:
            dummy = variables[0].__iter__
            # use as is
        except AttributeError:
            # add default cardinality of 2
            variables = [(variable_name, 2) for variable_name in variables]
        self.variables = variables
        self.variable_set = set(variable[0] for variable in variables)
        self.cardinalities = dict(variables)
        self.variable_to_axis = dict((variable[0], i)
                                     for i, variable in enumerate(variables))
        self.axis_to_variable = dict((i, variable[0])
                                     for i, variable in enumerate(variables))
        self.parameters = parameters
        if data is not None:
            if data.dtype == 'float64':
                self._data = data
            else:
                self._data = data.astype('float64')
        else:
            self._data = numpy.ones(tuple(variable[1] for variable in variables))

    def marginalize(self, variables_to_keep):
        """
        Marginalize a potential table.
        :param variables_to_keep: A list of variables to keep. All other variables in the factor are marginalized out.
        :returns: A new factor.
        """
        axes = [self.variable_to_axis[own_variable_name]
                for own_variable_name, _ in self.variables
                if own_variable_name not in variables_to_keep]
        result_variables = [variable for variable in self.variables if variable[0] in variables_to_keep]
        result_shape = [cardinality for _, cardinality in result_variables]

        if axes:
            result_data = numpy.sum(self._data, axis=tuple(axes)).reshape(result_shape)
        else:  # Edge case where the original array is returned instead of a copy
            result_data = numpy.copy(self._data)

        result_data = result_data
        result_factor = DiscreteFactor(result_variables, result_data)

        return result_factor

    def get_potential(self, variable_list):
        """
        Return the entries in the table for the assignment of variables.
        :param variable_list: List of (variable_name, assignment) tuples.
        """
        array_position = [slice(self.cardinalities[self.axis_to_variable[axis]])
                          for axis in xrange(len(self.variables))]
        for var, assignment in variable_list:
            if var in self.cardinalities:
                array_position[self.variable_to_axis[var]] = assignment
        return self._data[tuple(array_position)]

    def rotate_other(self, other_factor):
        """
        Helper to rotate another factor's potential table so that the variables it shares with this factor are along
        the same axes.
        :param other_factor: The other factor.
        """
        other_variable_order = [other_factor.axis_to_variable[other_axis]
                                for other_axis in xrange(len(other_factor.data.shape))]
        new_axis_order = [other_variable_order.index(self.axis_to_variable[axis])
                          for axis in xrange(len(other_variable_order))]
        return other_factor.data.transpose(new_axis_order)

    @property
    def data(self):
        """
        The normalized factor potentials.
        :returns: Potential table.
        """
        return self._data

    @property
    def log_data(self):
        """
        The log of the normalized factor potentials.
        :returns: Potential table.
        """
        return numpy.log(self._data)

    @property
    def normalized_data(self):
        """
        The potential table normalized so that the entries sum to one.
        :returns: Potential table.
        """
        return self._data / numpy.sum(self._data)

    def __str__(self):
        """
        For debugging, a factor is represented by its variables.
        :returns: String representation of the factor.
        """
        return 'F{' + ', '.join(str(var) for var, card in self.variables) + '}'

    def __repr__(self):
        """
        For debugging, the factor is represented by its variables.
        :returns: String representation of the factor.
        """
        return self.__str__()


class DiscreteBelief(DiscreteFactor):
    """
    A factor containing only discrete variables. Beliefs are mutable and contains the current belief about
    its random variables and metadata.
    """

    def __init__(self, factor=None, variables=None, data=None, parameters=None):
        """
        Constructor.

        :param factor: A factor to use to initialize the belief.
        """
        if factor:
            super(DiscreteBelief, self).__init__(factor.variables,
                                                 factor.data.copy(),
                                                 factor.parameters)
        else:
            super(DiscreteBelief, self).__init__(variables, data, parameters)

        self.evidence = {}

    def set_evidence(self, evidence):
        """
        Pin the variables to certain values - reducing the factor.
        :param evidence: Dictionary where the key is the variable name and the value its value.
        """
        # TODO: At the moment the factor is reduced by simply setting all unobserved values in the table to zero.
        #       Find a more efficient way of doing this.
        self.evidence = evidence
        array_position = [slice(self.cardinalities[self.axis_to_variable[axis]])
                          for axis in xrange(len(self.variables))]
        for var, assignment in evidence.items():
            if var in self.cardinalities:
                array_position[self.variable_to_axis[var]] = assignment

        multiplier = numpy.zeros_like(self._data)
        multiplier[tuple(array_position)] = 1
        self._data = self._data * multiplier * 1.0

    def set_parameters(self, parameters):
        """
        Fill the potential with exponential of the parameters.
        :param parameters: Dictionary where the key is a parameter name and the value the log value of the parameter.
        """
        original_shape = self._data.shape
        if self.parameters is not None:
            new_data = self._data.reshape(-1, )
            for i, parameter in enumerate(self.parameters.reshape(-1, )):
                if isinstance(parameter, str):
                    new_data[i] = numpy.exp(parameters[parameter])
                else:
                    new_data[i] = parameter
            self._data = new_data.reshape(original_shape)

    def normalize(self):
        """
        Update the potential table so that the entries sum to one.
        """
        self._data = self._data / numpy.sum(self._data)

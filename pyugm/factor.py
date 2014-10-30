"""
Module containing the factor classes.
"""
# License: BSD 3 clause

import numpy
from numba import jit  # void, f8, i1, b1, njit, jit


class DiscreteFactor(object):
    """
    A factor containing only discrete variables.
    """

    def __init__(self, variables, data=None, log_normalizer=0.0, parameters=None):
        """
        Constructor.

        :param variables: A list of (variable_name, cardinality) tuples, where variable_name is a string or int.
            If only a list of names is provided, cardinalities are assumed to be 2.
        :param data: ndarray of factor potentials.
        :param log_normalizer: The log of the factor with which the whole potential is multiplied.
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
            self._data = data
        else:
            self._data = numpy.ones(tuple(variable[1] for variable in variables))

        self._log_normalizer = log_normalizer

    def marginalize(self, variables_to_keep):
        """
        Marginalize a potential table.
        :param variables_to_keep: A list of variables to keep. All other variables in the factor are marginalized out.

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

        result_log_norm = numpy.log(result_data.sum())
        result_data = result_data / numpy.exp(result_log_norm)
        total_log_norm = self._log_normalizer + result_log_norm
        result_factor = DiscreteFactor(result_variables, result_data, log_normalizer=total_log_norm)

        return result_factor

    def multiply(self, other_factor, divide=False):
        """
        Multiply two factors.
        :param other_factor: The other factor to multiply into this factor.
        :param divide: If true then the other factor is divided into this factor, otherwise multiplied.
        """
        # pylint: disable=protected-access
        dim1 = len(other_factor.variables)
        dim2 = len(self.variables)
        strides1 = numpy.array(other_factor._data.strides, dtype=numpy.int8) / other_factor._data.itemsize
        strides2 = numpy.array(self._data.strides, dtype=numpy.int8) / self._data.itemsize
        card2 = numpy.array([self.cardinalities[self.axis_to_variable[axis]] for axis in xrange(dim2)],
                            dtype=numpy.int8)
        assignment1 = numpy.zeros(dim1, dtype=numpy.int8)
        assignment2 = numpy.zeros(dim2, dtype=numpy.int8)
        data1_flatshape = (numpy.prod(other_factor._data.shape),)
        data2_flatshape = (numpy.prod(self._data.shape),)
        variable1_to_2 = numpy.array([self.variable_to_axis[other_factor.axis_to_variable[ax1]]
                                      for ax1 in xrange(dim1)], dtype=numpy.int8)
        data1 = other_factor._data.view()
        data2 = self._data.view()
        data1.shape = data1_flatshape
        data2.shape = data2_flatshape
        multiply_factors(data1, data2,
                         strides1, strides2,
                         card2,
                         assignment1, assignment2,
                         variable1_to_2, divide)
        if divide:
            self._log_normalizer -= other_factor._log_normalizer
        else:
            self._log_normalizer += other_factor._log_normalizer

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
        return self._data[tuple(array_position)] * numpy.exp(self._log_normalizer)

    def set_evidence(self, evidence):
        """
        Pin the variables to certain values - reducing the factor.
        :param evidence: Dictionary where the key is the variable name and the value its value.
        """
        # TODO: At the moment the factor is reduced by simply setting all unobserved values in the table to zero.
        #       Find a more efficient way of doing this.
        array_position = [slice(self.cardinalities[self.axis_to_variable[axis]])
                          for axis in xrange(len(self.variables))]
        for var, assignment in evidence.items():
            if var in self.cardinalities:
                array_position[self.variable_to_axis[var]] = assignment

        multiplier = numpy.zeros_like(self._data)
        multiplier[tuple(array_position)] = 1
        self._data = self._data * multiplier * 1.0

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
    def log_normalizer(self):
        """
        The log of the normalizing factor for the potential.
        :returns: The log value.
        """
        return numpy.log(numpy.sum(self._data)) + self._log_normalizer

    @property
    def data(self):
        """
        The normalized factor potentials.
        :returns: Potential table.
        """
        return self._data * numpy.exp(self._log_normalizer)

    @property
    def log_data(self):
        """
        The log of the normalized factor potentials.
        :returns: Potential table.
        """
        return numpy.log(self._data) + self._log_normalizer

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


@jit  # (void(f8[:], f8[:], i1[:], i1[:], i1[:], i1[:], i1[:], i1[:], b1))
def multiply_factors(data1, data2,
                     strides1, strides2,
                     cardinalities2,
                     assignment1, assignment2,
                     variable1_to_2, divide):
    """
    Fast inplace factor multiplication.

    :param data1: Array to multiply in.
    :param data2: The larger array, containing all the variables in `data1` and also others.
    :param strides1: Stride array for `data1`.
    :param strides2: Stride array for `data2`.
    :param cardinalities2: Cardinalities of variables in `data2`.
    :param assignment1: A Numpy array with the same length as `data1`. Used as internal counter.
    :param assignment2: A Numpy array with the same length as `data2`. Used as internal counter.
    :param variable1_to_2: Permutation array where `variable1_to_2[i]` gives the index in `data2` of the variable `i` in
        `data1`.
    :param divide: Boolean - divides `data2` by `data1` if True, otherwise multiplies.
    """
    # pylint: disable=too-many-arguments
    # TODO: This is still quite slow - think about moving the complete update code to infer.py and to C or Cython.
    # Clear assignments
    for var1_i in range(len(assignment1)):
        assignment1[var1_i] = 0
    for var2_i in range(len(assignment2)):
        assignment2[var2_i] = 0
    done = False

    while not done:
        # Assign first from second assignment
        for var1_i in range(len(assignment1)):
            assignment1[var1_i] = assignment2[variable1_to_2[var1_i]]
        # Get indices in data
        assignment1_index = 0
        for var1_i in range(len(strides1)):
            assignment1_index += strides1[var1_i] * assignment1[var1_i]
        assignment2_index = 0
        for var2_i in range(len(strides2)):
            assignment2_index += strides2[var2_i] * assignment2[var2_i]
        # Multiply
        if not divide:
            data2[assignment2_index] *= data1[assignment1_index]
        else:
            if data2[assignment2_index] > 0.0:
                data2[assignment2_index] /= data1[assignment1_index]

        # Tick variable2 assignment
        assignment2[0] += 1
        for var2_i in range(len(assignment2) - 1):
            if assignment2[var2_i] >= cardinalities2[var2_i]:
                assignment2[var2_i] = 0
                assignment2[var2_i + 1] += 1
        if assignment2[-1] >= cardinalities2[-1]:
            done = True

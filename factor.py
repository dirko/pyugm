import numpy as np
from numba import jit, void, f8, i1, b1, njit


#@njit
@njit(void(f8[:], f8[:], i1[:], i1[:], i1[:], i1[:], i1[:], i1[:], b1))
def multiply_factors(data1, data2,
                     strides1, strides2,
                     card2,
                     assignment1, assignment2,
                     variable1_to_2, divide):
    """
    Fast inplace factor multiplication. data2 is the larger array, containing all the variables in data1 and also
    others.
    """
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
            if assignment2[var2_i] >= card2[var2_i]:
                assignment2[var2_i] = 0
                assignment2[var2_i + 1] += 1
        if assignment2[-1] >= card2[-1]:
            done = True


class DiscreteFactor:
    # noinspection PyNoneFunctionAssignment
    def __init__(self, variables, data=None, normalize=False, parameters=None):
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
            self.data = data
        else:
            self.data = np.ones(tuple(variable[1] for variable in variables))

        self.log_normalizer = np.log(1.0)
        if normalize:
            self.log_normalizer = np.log(self.data.sum())
            self.data /= np.exp(self.log_normalizer)
            self.log_normalizer = np.log(1.0)

        # Caching
        self.cached_multiply = {}

    def marginalize(self, variable_names, normalize=False):
        # variable_names: variables to keep
        #print 'marg', self.data
        axes = [self.variable_to_axis[own_variable_name]
                for own_variable_name, own_variable_card in self.variables
                if own_variable_name not in variable_names]
        result_variables = [variable for variable in self.variables if variable[0] in variable_names]
        result_shape = [cardinality for name, cardinality in result_variables]

        if axes:
            #result_data = np.apply_over_axes(np.sum, self.data, axes).reshape(result_shape)
            #print axes, result_shape
            result_data = np.sum(self.data, axis=tuple(axes)).reshape(result_shape)
        else:  # Edge case where the original array is returned instead of a copy
            result_data = np.copy(self.data)

        result_log_norm = np.log(result_data.sum())
        #print 'susm', result_log_norm, result_data, self.data
        result_data = result_data / np.exp(result_log_norm)
        #print 'after exp div', result_data
        total_log_norm = self.log_normalizer + result_log_norm
        if normalize:
            total_log_norm = np.log(1.0)
        result_factor = DiscreteFactor(result_variables, result_data)
        result_factor.log_normalizer = total_log_norm

        return result_factor

    def multiply(self, other_factor, divide=False):
        dim1 = len(other_factor.variables)
        dim2 = len(self.variables)
        strides1 = np.array(other_factor.data.strides, dtype=np.int8) / other_factor.data.itemsize
        strides2 = np.array(self.data.strides, dtype=np.int8) / self.data.itemsize
        card2 = np.array([self.cardinalities[self.axis_to_variable[axis]] for axis in xrange(dim2)], dtype=np.int8)
        assignment1 = np.zeros(dim1, dtype=np.int8)
        assignment2 = np.zeros(dim2, dtype=np.int8)
        data1_flatshape = (np.prod(other_factor.data.shape),)
        data2_flatshape = (np.prod(self.data.shape),)
        variable1_to_2 = np.array([self.variable_to_axis[other_factor.axis_to_variable[ax1]] for ax1 in xrange(dim1)], dtype=np.int8)
        data1 = other_factor.data.view()
        data2 = self.data.view()
        data1.shape = data1_flatshape
        data2.shape = data2_flatshape
        multiply_factors(data1, data2,
                         strides1, strides2,
                         card2,
                         assignment1, assignment2,
                         variable1_to_2, divide)
        if divide:
            self.log_normalizer -= other_factor.log_normalizer
        else:
            self.log_normalizer += other_factor.log_normalizer

    def get_potential(self, variable_list):
        """
        Return the entry in the table for the assignment of variables.
        variable_list: list of pairs. Each pair is (variable_name, assignment)
        """
        array_position = [slice(self.cardinalities[self.axis_to_variable[axis]])
                          for axis in xrange(len(self.variables))]
        for var, assignment in variable_list:
            if var in self.cardinalities:
                array_position[self.variable_to_axis[var]] = assignment
        return self.data[tuple(array_position)] * np.exp(self.log_normalizer)

    def set_evidence(self, evidence, normalize=False, inplace=False):
        """
        Pin the variables to certain values
        param evidence: list of (variable, value) pairs
        return: new factor if inplace is True, else this factor.
        """
        array_position = [slice(self.cardinalities[self.axis_to_variable[axis]])
                          for axis in xrange(len(self.variables))]
        for var, assignment in evidence.items():
            if var in self.cardinalities:
                array_position[self.variable_to_axis[var]] = assignment

        multiplier = np.zeros_like(self.data)
        multiplier[tuple(array_position)] = 1

        if inplace:
            return_factor = self
        else:
            return_factor = DiscreteFactor(self.variables, 'placeholder', parameters=self.parameters)
        return_data = self.data * multiplier * 1.0

        total_log_norm = self.log_normalizer #+ np.log(result_norm)
        if normalize:
            result_norm = return_data.sum()
            return_data /= result_norm
            total_log_norm = np.log(1.0)
        return_factor.data = return_data
        return_factor.log_normalizer = total_log_norm
        #print 'set_ev', self.variables, return_data
        return return_factor

    def _rotate_other(self, other_factor):
        other_variable_order = [other_factor.axis_to_variable[other_axis]
                                for other_axis in xrange(len(other_factor.data.shape))]
        new_axis_order = [other_variable_order.index(self.axis_to_variable[axis])
                          for axis in xrange(len(other_variable_order))]
        #print self.variables
        #print other_factor.variables
        #print other_variable_order
        #print new_axis_order
        return other_factor.data.transpose(new_axis_order)

    def get_data(self):
        """ Return factor potential. """
        return self.data * np.exp(self.log_normalizer)

    def __str__(self):
        return 'F{' + ', '.join(str(var) for var, card in self.variables) + '}'

    def __repr__(self):
        return self.__str__()

import numpy as np


class Factor:
    def __init__(self, variables, data=None):
        # variables = [(name, cardinality), (name, cardinality) ... ]

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
        if data is not None:
            self.data = data
        else:
            self.data = np.ones(tuple(variable[1] for variable in variables))

    def marginalize(self, variable_names):
        # variable_names: variables to keep
        axes = [self.variable_to_axis[own_variable_name]
                for own_variable_name, own_variable_card in self.variables
                if own_variable_name not in variable_names]
        result_variables = [variable for variable in self.variables if variable[0] in variable_names]
        result_shape = [cardinality for name, cardinality in result_variables]

        result_data = np.apply_over_axes(np.sum, self.data, axes).reshape(result_shape)
        result_factor = Factor(result_variables, result_data)

        return result_factor

    def multiply(self, other_factor, divide=False):
        # print self.data.shape
        # print other_factor.axis_to_variable
        # print self.axis_to_variable
        # print [other_factor.axis_to_variable[other_axis] for other_axis in xrange(len(other_factor.data.shape))]
        other_variable_order = [other_factor.axis_to_variable[other_axis]
                                for other_axis in xrange(len(other_factor.data.shape))]
        variables_in_self_not_in_other = [variable[0] for variable in self.variables
                                          if variable not in other_factor.variables]
        other_variable_order += variables_in_self_not_in_other
        new_axis_order = [self.variable_to_axis[other_variable] for other_variable in other_variable_order]

        new_shape = [card for card in other_factor.data.shape] + [1 for var in variables_in_self_not_in_other]
        reshaped_other_data = other_factor.data.reshape(new_shape)
        reordered_other_data = reshaped_other_data.transpose(new_axis_order)
        # print 'new_axis_order', new_axis_order
        # print 'reshaped_other_data.shape', reshaped_other_data.shape
        # print 'reordered_other_data.shape', reordered_other_data.shape

        # print 'zipped shapes', [s for s in zip(self.data.shape, reordered_other_data.shape)]
        tile_shape = [self_card if self_card != other_card else 1
                      for self_card, other_card in zip(self.data.shape, reordered_other_data.shape)]
        # print 'tile_shape', tile_shape
        tiled_other_data = np.tile(reordered_other_data, tile_shape)
        # print 'tile_other_data', tiled_other_data
        # print 'res'
        # print self.data * tiled_other_data
        result_factor = Factor(self.variables, 'placeholder')
        if not divide:
            result_factor.data = self.data * tiled_other_data
        else:
            result_factor.data = self.data / tiled_other_data
        return result_factor

    def __str__(self):
        return '{' + ', '.join(str(var) for var, card in self.variables) + '}'
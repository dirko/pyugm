import numpy as np


class Model:
    def __init__(self):
        self.factors = []
        self.cardinalities = dict()

    def add_factor(self, factor):
        self.factors.append(factor)

        for variable in factor.variables:
            if variable[0] in self.cardinalities:
                assert(variable[1] == self.cardinalities[variable[0]])
            else:
                self.cardinalities[variable[0]] = variable[1]


class Factor:
    def __init__(self, variables, data=None):
        # variables = [(name, cardinality), (name, cardinality) ... ]

        self.variables = variables
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
        axes = [self.variable_to_axis[variable_name] for variable_name in variable_names]
        result_variables = [(name, self.cardinalities[name]) for name in variable_names]

        result_data = np.apply_over_axes(np.sum, self.data, axes)
        result_factor = Factor(result_variables, result_data)

        return result_factor

    def multiply(self, other_factor):
        pass


class Inference:

    def __init__(self, factor_list):
        pass

    def run(self):
        pass

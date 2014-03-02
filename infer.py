import numpy as np


class Model:
    def __init__(self, factor_list):
        self.factors = []
        self.cardinalities = dict()
        self.variables_to_factors = dict()
        self.edges = []

        for factor in factor_list:
            self.add_factor(factor)

    def add_factor(self, factor):
        self.factors.append(factor)

        for variable in factor.variables:
            if variable[0] in self.cardinalities:
                assert(variable[1] == self.cardinalities[variable[0]])
            else:
                self.cardinalities[variable[0]] = variable[1]

            if variable[0] in self.variables_to_factors:
                self.variables_to_factors[variable[0]].add(factor)
            else:
                self.variables_to_factors[variable[0]] = {factor}

    def build_graph(self):
        # Build rest of graph by greedily adding the largest sepset factors to the above added node
        for variable, factors in self.variables_to_factors.items():
            marked_factors = set()
            unmarked_factors = set(factors)

            if len(factors) > 1:
                first_candidate_sepset = self.get_largest_unmarked_sepset(variable,
                                                                          list(factors),
                                                                          unmarked_factors,  # just for start
                                                                          unmarked_factors)
                self.edges += [(first_candidate_sepset[0], first_candidate_sepset[1])]
                marked_factors.add(first_candidate_sepset[0])
                marked_factors.add(first_candidate_sepset[1])
                unmarked_factors.remove(first_candidate_sepset[0])
                unmarked_factors.remove(first_candidate_sepset[1])

                while len(marked_factors) < len(factors):
                    largest_sepset = self.get_largest_unmarked_sepset(variable, list(factors), marked_factors,
                                                                      unmarked_factors)
                    # Add largest sepset factors to graph
                    if largest_sepset[0] is not None:
                        self.edges += [(largest_sepset[0], largest_sepset[1])]
                        marked_factors.add(largest_sepset[0])
                        marked_factors.add(largest_sepset[1])
                        if largest_sepset[0] in unmarked_factors:
                            unmarked_factors.remove(largest_sepset[0])
                        if largest_sepset[1] in unmarked_factors:
                            unmarked_factors.remove(largest_sepset[1])

    def get_largest_unmarked_sepset(self, variable, factors, marked_factors, unmarked_factors):
        sepset_sizes = [(factor1, factor2, len(factor1.variable_set.intersection(factor2.variable_set)))
                        for factor1 in factors for factor2 in factors
                        if factor1 in marked_factors and factor2 in unmarked_factors and
                        factor1 != factor2 and
                        variable in factor1.variable_set and variable in factor2.variable_set]
        if len(sepset_sizes) == 0:
            max_sepset = (None, None, 0)
        else:
            max_sepset = max(sepset_sizes, key=lambda x: x[2])
        return max_sepset

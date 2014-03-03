import numpy as np


class Model:
    def __init__(self, factor_list):
        self.factors = []
        self.cardinalities = dict()
        self.variables_to_factors = dict()
        self.edges = set()

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

        def add_edge_to_set(edge, set_to_add):
            # Edge should be undirected - but we have to use tuples (need to be hashable)
            if (edge[1], edge[0]) not in set_to_add:
                set_to_add.add(edge)

        def mark_factors_in_edge(edge, marked_set, unmarked_set):
            marked_set.add(edge[0])
            marked_set.add(edge[1])
            if edge[0] in unmarked_set:
                unmarked_set.remove(edge[0])
            if edge[1] in unmarked_set:
                unmarked_set.remove(edge[1])

        # Build graph by greedily adding the largest sepset factors to the above added node
        for variable, factors in self.variables_to_factors.items():
            marked_factors = set()
            unmarked_factors = set(factors)

            if len(factors) > 1:
                first_candidate_sepset = self.get_largest_unmarked_sepset(variable,
                                                                          list(factors),
                                                                          unmarked_factors,  # just for start
                                                                          unmarked_factors)
                add_edge_to_set(first_candidate_sepset, self.edges)
                mark_factors_in_edge(first_candidate_sepset, marked_factors, unmarked_factors)

                while len(marked_factors) < len(factors):
                    largest_sepset = self.get_largest_unmarked_sepset(variable, list(factors), marked_factors,
                                                                      unmarked_factors)
                    # Add largest sepset factors to graph
                    if largest_sepset is not None:
                        add_edge_to_set(largest_sepset, self.edges)
                        mark_factors_in_edge(largest_sepset, marked_factors, unmarked_factors)

    def get_largest_unmarked_sepset(self, variable, factors, marked_factors, unmarked_factors):
        sepset_sizes = [((factor1, factor2), len(factor1.variable_set.intersection(factor2.variable_set)))
                        for factor1 in factors for factor2 in factors
                        if factor1 in marked_factors and factor2 in unmarked_factors and
                        factor1 != factor2 and
                        variable in factor1.variable_set and variable in factor2.variable_set]
        if len(sepset_sizes) == 0:
            max_sepset = (None, 0)
        else:
            max_sepset = max(sepset_sizes, key=lambda x: x[1])
        return max_sepset[0]

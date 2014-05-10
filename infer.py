from factor import Factor
from Queue import PriorityQueue, Queue
import numpy as np


class Model:
    def __init__(self, factor_list):
        self.factors = []  # factor objects
        self.cardinalities = dict()  # variable name to int
        self.variables_to_factors = dict()  # variable name to factor
        self.edges = set()  # pairs of factors

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

    def set_evidence(self, evidence, normalize=False):
        """ param evidence: list of (variable name, value) pairs """
        for variable, value in evidence:
            for factor in self.variables_to_factors[variable]:
                factor.set_evidence([(variable, value)], normalize=normalize, inplace=True)


class LoopyBeliefUpdateInference:
    def __init__(self, model):
        self.separator_potentials = dict()  # edge to pairs of separator potentials
        self.belief_update_queue = PriorityQueue()  # edges to update beliefs for
        self.model = model

    def set_up_belief_update(self):
        """ Initialise separator potentials to 1 """
        for edge in list(self.model.edges):
            factor1, factor2 = edge
            sepset = factor1.variable_set.intersection(factor2.variable_set)
            separator_variables = [(variable, factor1.cardinalities[variable]) for variable in list(sepset)]
            # NOTE: will want to set this up more generically for other kinds of factors
            separator_factors = (Factor(separator_variables), Factor(separator_variables))
            self.separator_potentials[edge] = separator_factors
            self.separator_potentials[(edge[1], edge[0])] = separator_factors

        # Sets up priority queue of edges to update
        # NOTE: Should be possible to set up scheme here so that if the graph is a tree only a forward
        # and backward pass will be necessary.
        for edge in list(self.model.edges):
            priority = -np.inf  # highest priority
            self.belief_update_queue.put((priority, edge))

    def update_belief(self, edge):
        old_separators = self.separator_potentials[edge]
        variables_to_keep = old_separators[0].variable_set

        # Phi** = Sum Psi
        new_separator = edge[0].marginalize(variables_to_keep)
        # A = Phi* / Phi
        new_separator_divided = new_separator.multiply(old_separators[0], divide=True, update_inplace=False)
        # Psi** = Psi* x A
        print edge[0], '->', edge[1]
        print new_separator.data, '/', old_separators[0].data, '=', new_separator_divided.data
        edge[1].multiply(new_separator_divided)

        new_separators = (new_separator, old_separators[0])
        reverse_edge = (edge[1], edge[0])
        self.separator_potentials[edge] = new_separators
        self.separator_potentials[reverse_edge] = new_separators

        num_cells = np.prod(new_separator.data.shape) * 1.0
        average_change_per_cell = abs(new_separator.data - old_separators[0].data).sum() / num_cells
        print average_change_per_cell
        print new_separator.data

        return average_change_per_cell

    def update_beliefs(self, number_of_updates=100, number_of_zero_updates_before_complete_pass=2, delta=10**-10):
        total_average_change_per_cell = 0
        number_of_consecutive_zero_updates = 0
        update_num = 0
        for update_num in xrange(number_of_updates):
            print '-------new update------', update_num
            priority, edge = self.belief_update_queue.get_nowait()
            self.belief_update_queue.task_done()

            average_change_per_cell = self.update_belief(edge)
            total_average_change_per_cell += average_change_per_cell

            # Update queue
            reverse_edge = (edge[1], edge[0])
            self.belief_update_queue.put((-average_change_per_cell, reverse_edge))

            if average_change_per_cell == 0:
                number_of_consecutive_zero_updates += 1
            else:
                number_of_consecutive_zero_updates = 0

            if number_of_consecutive_zero_updates == number_of_zero_updates_before_complete_pass:
                edge_list = list(self.model.edges)
                complete_pass_change = 0
                for edge in edge_list:
                    complete_pass_change += self.update_belief(edge)
                edge_list.reverse()
                for edge in edge_list:
                    reverse_edge = (edge[1], edge[0])
                    complete_pass_change += self.update_belief(reverse_edge)
                if complete_pass_change < delta:
                    break
                else:
                    number_of_consecutive_zero_updates = 0

        return total_average_change_per_cell / number_of_updates, update_num

    def exhaustive_enumeration(self):
        """ Compute the complete probability table by enumerating all variable instantiations """
        variables = [(key, value) for key, value in self.model.cardinalities.items()]
        table_shape = [cardinality for cardinality, variable_name in variables]
        table_size = np.prod(table_shape)
        if table_size > 10**7:
            raise Exception('Model too large for exhaustive enumeration')

        new_factor = Factor(variables)
        instantiation = [[var[0], 0] for var in variables]

        def tick_instantiation(i):
            """ Return the next instantiation given previous one i """
            i[0][1] += 1
            i_done = False
            for variable_counter in xrange(len(i) - 1):
                if i[variable_counter][1] >= variables[variable_counter][1]:  # Instantiation > than cardinality
                    i[variable_counter][1] = 0
                    i[variable_counter + 1][1] += 1
            if i[-1][1] >= variables[-1][1]:
                i_done = True
            return i_done, i

        done = False
        while not done:
            for factor in self.model.factors:
                potential_value = factor.get_potential([tuple(var) for var in instantiation])
                new_factor.data[tuple(var[1] for var in instantiation)] *= potential_value
            done, instantiation = tick_instantiation(instantiation)

        return new_factor

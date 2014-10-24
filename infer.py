"""
Module containing the inference routines.
"""
from factor import DiscreteFactor
import numpy as np


class FloodingProtocol:
    """
    Defines an update ordering
    """
    def __init__(self, model, max_iterations=np.inf, converge_delta=10**-10):
        self._model = model
        self._edges = list(model.edges)
        self._current_edge_index = 0
        self.total_iterations = 0
        self._max_iterations = max_iterations
        self.current_iteration_delta = 0
        self._converge_delta = converge_delta

    def reset(self):
        self._edges = list(self._model.edges)
        self._current_edge_index = 0
        self.total_iterations = 0
        self.current_iteration_delta = 0

    def next_edge(self, last_update_change):
        """ Get the next edge to update """
        self.current_iteration_delta += last_update_change
        if self._edges:
            next_edge = self._edges[self._current_edge_index / 2]
        else:
            return None
        if self._current_edge_index % 2 == 1:
            reversed_edge = (next_edge[1], next_edge[0])
            next_edge = reversed_edge

        self._current_edge_index += 1
        if self._current_edge_index / 2 >= len(self._edges):
            self._current_edge_index = 0
            self.total_iterations += 1
            if self.total_iterations > self._max_iterations or self.current_iteration_delta < self._converge_delta:
                next_edge = None
            self.current_iteration_delta = 0.0
        return next_edge


class DistributeCollectProtocol(object):
    """ Works only for trees """
    def __init__(self, model):
        self._model = model
        self._edges = model.edges
        self._to_visit = set()
        self._visited_factors = set()
        self._forward_edges = []
        self.current_iteration_delta = 0.0
        self.total_iterations = 0
        for sub_graph in self._model.disconnected_subgraphs:  # Roots
            root_factor = list(sub_graph)[0]
            self._visited_factors.add(root_factor)
            for edge in self._edges:
                if edge[0] == root_factor:
                    self._to_visit.add(edge[1])
                elif edge[1] == root_factor:
                    self._to_visit.add(edge[0])
        while len(self._to_visit) > 0:
            next_factor = self._to_visit.pop()
            next_edge = None
            for edge in self._edges:
                if edge[0] == next_factor and edge[1] in self._visited_factors:
                    next_edge = (edge[1], edge[0])
                    self._visited_factors.add(next_factor)
                elif edge[1] == next_factor and edge[0] in self._visited_factors:
                    next_edge = edge
                    self._visited_factors.add(next_factor)
                elif edge[0] == next_factor and edge[1] not in self._visited_factors:
                    self._to_visit.add(edge[1])
                elif edge[1] == next_factor and edge[0] not in self._visited_factors:
                    self._to_visit.add(edge[0])
            self._forward_edges.append(next_edge)

        reversed_edges = [(edge[1], edge[0]) for edge in self._forward_edges[::-1]]
        self._all_edges = self._forward_edges + reversed_edges
        self._all_edges += self._all_edges
        self._counter = 0

    def reset(self):
        self._counter = 0

    def next_edge(self, last_update_change):
        """ Get the next edge to update """
        self.current_iteration_delta += last_update_change
        if self._counter < len(self._all_edges):
            return_edge = self._all_edges[self._counter]
            self._counter += 1
            return return_edge
        else:
            return None


class LoopyBeliefUpdateInference:
    def __init__(self, model):
        self.separator_potential = dict()  # edge to pairs of separator potentials
        self.model = model

    def set_up_belief_update(self):
        """ Initialise separator potentials to 1 """
        for edge in list(self.model.edges):
            factor1, factor2 = edge
            sepset = factor1.variable_set.intersection(factor2.variable_set)
            separator_variables = [(variable, factor1.cardinalities[variable]) for variable in list(sepset)]
            # NOTE: will want to set this up more generically for other kinds of factors
            separator_factor = DiscreteFactor(separator_variables)
            self.separator_potential[edge] = separator_factor
            self.separator_potential[(edge[1], edge[0])] = separator_factor

    def update_belief(self, edge):
        old_separator = self.separator_potential[edge]
        variables_to_keep = old_separator.variable_set

        # Could probably be a bit faster
        new_separator = edge[0].marginalize(variables_to_keep)
        new_separator_divided = edge[0].marginalize(variables_to_keep)
        new_separator_divided.multiply(old_separator, divide=True)
        edge[1].multiply(new_separator_divided)

        reverse_edge = (edge[1], edge[0])
        self.separator_potential[edge] = new_separator
        self.separator_potential[reverse_edge] = new_separator

        num_cells = np.prod(new_separator._data.shape) * 1.0
        average_change_per_cell = abs(new_separator._data - new_separator._rotate_other(old_separator)).sum() / num_cells

        return average_change_per_cell

    def update_beliefs(self, update_order=None, number_of_updates=100):
        if not update_order:
            update_order = FloodingProtocol(self.model, max_iterations=number_of_updates)

        average_change_per_cell = 0
        edge = update_order.next_edge(average_change_per_cell)
        while edge:
            average_change_per_cell = self.update_belief(edge)
            edge = update_order.next_edge(average_change_per_cell)

        # Find normaliser
        total_z = 0.0
        for island in self.model.disconnected_subgraphs:
            total_z += np.log(np.sum(list(island)[0]._data)) + list(island)[0].log_normalizer
        # Multiply so each factor has the same normaliser
        for island in self.model.disconnected_subgraphs:
            island_z = np.log(np.sum(list(island)[0]._data)) + list(island)[0].log_normalizer
            for factor in list(island):
                factor.log_normalizer += (total_z - island_z)

        return update_order.current_iteration_delta, update_order.total_iterations

    def exhaustive_enumeration(self):
        """ Compute the complete probability table by enumerating all variable instantiations """
        variables = [(key, value) for key, value in self.model.cardinalities.items()]
        table_shape = [cardinality for cardinality, variable_name in variables]
        table_size = np.prod(table_shape)
        if table_size > 10**7:
            raise Exception('Model too large for exhaustive enumeration')

        new_factor = DiscreteFactor(variables)
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
                new_factor._data[tuple(var[1] for var in instantiation)] *= potential_value
            done, instantiation = tick_instantiation(instantiation)

        return new_factor

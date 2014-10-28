"""
Module containing the inference routines.
"""

import numpy as np

from pyugm.factor import DiscreteFactor


class LoopyBeliefUpdateInference:
    """
    An inference object to calibrate the potentials.
    """
    def __init__(self, model):
        """
        Constructor.
        :param model: The model.
        """
        self._separator_potential = dict()  # edge to pairs of separator potentials
        self._model = model
        self._set_up_belief_update()

    def _set_up_belief_update(self):
        """
        Helper to initialise separator potentials to 1.
        """
        for edge in list(self._model.edges):
            factor1, factor2 = edge
            sepset = factor1.variable_set.intersection(factor2.variable_set)
            separator_variables = [(variable, factor1.cardinalities[variable]) for variable in list(sepset)]
            # NOTE: will want to set this up more generically for other kinds of factors
            separator_factor = DiscreteFactor(separator_variables)
            self._separator_potential[edge] = separator_factor
            self._separator_potential[(edge[1], edge[0])] = separator_factor

    def _update_belief(self, edge):
        """
        Helper to update the beliefs along a single edge.
        :param edge: A tuple of two factors.
        """
        old_separator = self._separator_potential[edge]
        variables_to_keep = old_separator.variable_set

        # Could probably be a bit faster
        new_separator = edge[0].marginalize(variables_to_keep)
        new_separator_divided = edge[0].marginalize(variables_to_keep)
        new_separator_divided.multiply(old_separator, divide=True)
        edge[1].multiply(new_separator_divided)

        reverse_edge = (edge[1], edge[0])
        self._separator_potential[edge] = new_separator
        self._separator_potential[reverse_edge] = new_separator

        num_cells = np.prod(new_separator.data.shape) * 1.0
        average_change_per_cell = abs(new_separator.data - new_separator.rotate_other(old_separator)).sum() / num_cells

        return average_change_per_cell

    def calibrate(self, update_order=None):
        """
        Calibrate all the factors in the model by running belief updates according to the `update_order` ordering
        scheme.
        :param update_order: A message update protocol. If `None`, `FloodingProtocol` is used.
        :param number_of_updates: Number of times to update every edge if the default Flooding protocol is used.
        """
        if not update_order:
            update_order = FloodingProtocol(self._model)

        average_change_per_cell = 0
        edge = update_order.next_edge(average_change_per_cell)
        while edge:
            average_change_per_cell = self._update_belief(edge)
            edge = update_order.next_edge(average_change_per_cell)

        # Find normaliser
        total_z = 0.0
        for island in self._model.disconnected_subgraphs:
            total_z += list(island)[0].log_normalizer
        # Multiply so each factor has the same normaliser
        for island in self._model.disconnected_subgraphs:
            island_z = list(island)[0].log_normalizer
            for factor in list(island):
                factor._log_normalizer += (total_z - island_z)

        return update_order.current_iteration_delta, update_order.total_iterations


class ExhaustiveEnumeration:
    """
    A test inference object to build the complete potential table.
    """
    def __init__(self, model):
        """
        Constructor.
        :param model: The model.
        """
        self._model = model

    def exhaustively_enumerate(self):
        """
        Compute the complete probability table by enumerating all variable instantiations.
        :returns: A factor of all the variables in the original model.
        """
        variables = [(key, value) for key, value in self._model.cardinalities.items()]
        table_shape = [cardinality for cardinality, variable_name in variables]
        table_size = np.prod(table_shape)
        if table_size > 10**7:
            raise Exception('Model too large for exhaustive enumeration')

        new_factor = DiscreteFactor(variables)
        instantiation = [[var[0], 0] for var in variables]

        def _tick_instantiation(i_list):
            """
            Helper to give the next instantiation given the previous one.
            :param i_list: List of [variable_name, variable_value] lists.
            :returns: A tuple (i_done, i_list), where i_done is True if all instantiations of the variables have been
                        visited and False otherwise, and i_list is a new instantiation.
            """
            i_list[0][1] += 1
            i_done = False
            for variable_counter in xrange(len(i_list) - 1):
                if i_list[variable_counter][1] >= variables[variable_counter][1]:  # Instantiation > than cardinality
                    i_list[variable_counter][1] = 0
                    i_list[variable_counter + 1][1] += 1
            if i_list[-1][1] >= variables[-1][1]:
                i_done = True
            return i_done, i_list

        done = False
        while not done:
            for factor in self._model.factors:
                potential_value = factor.get_potential([tuple(var) for var in instantiation])
                new_factor._data[tuple(var[1] for var in instantiation)] *= potential_value
            done, instantiation = _tick_instantiation(instantiation)

        return new_factor


class FloodingProtocol:
    """
    Defines an update ordering where updates are done in both directions for each edge in the cluster graph.
    """
    def __init__(self, model, max_iterations=20, converge_delta=10**-10):
        """
        Constructor.
        :param model: The model.
        :param max_iterations: Maximum number of times to update each edge.
        :param converge_delta: Stop updates after the change in potential between two passes over all the
                        edges is less than `converge_delta`.
        """
        self.total_iterations = 0
        self.current_iteration_delta = 0
        self._model = model
        self._edges = list(model.edges)
        self._current_edge_index = 0
        self._max_iterations = max_iterations
        self._converge_delta = converge_delta

    def reset(self):
        """
        Reset the object.
        """
        self.total_iterations = 0
        self.current_iteration_delta = 0
        self._edges = list(self._model.edges)
        self._current_edge_index = 0

    def next_edge(self, last_update_change):
        """
        Get the next edge to update.
        :param last_update_change: The average factor-potential change that the previous update caused. Used to compute
                        convergence.
        :returns: An edge.
        """
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
    """
    Defines an update ordering where an edge is only updated if all other edges to the source node has already been
    updated. Works only for trees.
    """
    def __init__(self, model):
        """
        Constructor.
        :param model: The model.
        """
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
        """
        Reset the object.
        """
        self._counter = 0

    def next_edge(self, last_update_change):
        """
        Get the next edge to update.
        :param last_update_change: Change in factor-potential that the previous update caused.
        :returns: An edge.
        """
        self.current_iteration_delta += last_update_change
        if self._counter < len(self._all_edges):
            return_edge = self._all_edges[self._counter]
            self._counter += 1
            return return_edge
        else:
            return None

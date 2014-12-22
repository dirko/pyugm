"""
Module containing the inference routines.
"""
# License: BSD 3 clause

import numpy

from pyugm.factor import DiscreteFactor


class LoopyBeliefUpdateInference(object):
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

    def _update_belief(self, edge, normalize=False):
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
        if normalize:
            edge[1].normalize()

        reverse_edge = (edge[1], edge[0])
        self._separator_potential[edge] = new_separator
        self._separator_potential[reverse_edge] = new_separator

        num_cells = numpy.prod(new_separator.data.shape) * 1.0
        average_change_per_cell = abs(new_separator.data - new_separator.rotate_other(old_separator)).sum() / num_cells

        return average_change_per_cell

    def calibrate(self, update_order=None):
        """
        Calibrate all the factors in the model by running belief updates according to the `update_order` ordering
        scheme.
        :param update_order: A message update protocol. If `None`, `FloodingProtocol` is used.
        """
        if not update_order:
            update_order = FloodingProtocol(self._model)

        average_change_per_cell = 0
        edge = update_order.next_edge(average_change_per_cell)
        while edge:
            average_change_per_cell = self._update_belief(edge, normalize=True)
            edge = update_order.next_edge(average_change_per_cell)

        return update_order.current_iteration_delta, update_order.total_iterations


class TreeBeliefUpdateInference(LoopyBeliefUpdateInference):
    """
    An inference object to calibrate the potentials. Because exact inference is possible, we can easily find
    the partition function in this case.
    """

    def calibrate(self, update_order=None):
        """
        Calibrate all the factors in the model by running belief updates according to the `update_order` ordering
        scheme.
        :param update_order: A message update protocol. If `None`, `FloodingProtocol` is used.
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


class ExhaustiveEnumeration(object):
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
        table_shape = [cardinality for _, cardinality in variables]
        table_size = numpy.prod(table_shape)
        if table_size > 10**7:
            raise Exception('Model too large for exhaustive enumeration')

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

        data = numpy.ones(table_shape)
        done = False
        while not done:
            for factor in self._model.factors:
                potential_value = factor.get_potential([tuple(var) for var in instantiation])
                data[tuple(var[1] for var in instantiation)] *= potential_value
            done, instantiation = _tick_instantiation(instantiation)

        new_factor = DiscreteFactor(variables, data=data)
        return new_factor


class FloodingProtocol(object):
    """
    Defines an update ordering where updates are done in both directions for each edge in the cluster graph.
    """
    def __init__(self, model, max_iterations=20, converge_delta=10**-10, callback=None):
        """
        Constructor.
        :param model: The model.
        :param max_iterations: Maximum number of times to update each edge.
        :param converge_delta: Stop updates after the change in potential between two passes over all the
                        edges is less than `converge_delta`.
        :callback: Function to call at the end of every iteration.
        """
        self.total_iterations = 0
        self.current_iteration_delta = 0
        self._model = model
        self._edges = list(model.edges)
        self._current_edge_index = 0
        self._max_iterations = max_iterations
        self._converge_delta = converge_delta
        self._callback = callback

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
            numpy.random.shuffle(self._edges)
            self.total_iterations += 1
            if self._callback:
                self._callback(self)
            if self.total_iterations > self._max_iterations or self.current_iteration_delta < self._converge_delta:
                next_edge = None
            else:
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


class LoopyDistributeCollectProtocol(object):
    """
    Defines an update ordering where an edge is only updated if all other edges to the source node has already been
    updated. At each iteration, a different 'tree' on the complete (loopy) graph is traversed.
    """
    def __init__(self, model, max_iterations=20, converge_delta=10**-10):
        """
        Constructor.
        :param model: The model.
        :param max_iterations: The number of time to update each edge.
        :param converge_delta: If the potential table difference between the two iterations is less than this number,
                               updates will stop.
        """
        self._model = model
        self._edges = model.edges
        self.current_iteration_delta = 0.0
        self.total_iterations = 0
        self._all_edges = []
        self._counter = 0
        self._max_iterations = max_iterations
        self._converge_delta = converge_delta

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
        if self._counter >= len(self._all_edges):
            #if self.current_iteration_delta < self._converge_delta or self.total_iterations > self._max_iterations:
            if self.total_iterations > self._max_iterations:
                return None
            self._build_tree()
            self._counter = 0
            self.current_iteration_delta = 0.0
            self.total_iterations += 1
        return_edge = self._all_edges[self._counter]
        self._counter += 1
        return return_edge

    def _build_tree(self):
        """
        Helper to generate a new ordering
        """
        _to_visit = set()
        _visited_factors = set()
        _forward_edges = []
        for sub_graph in self._model.disconnected_subgraphs:  # Roots
            root_factor = list(sub_graph)[numpy.random.randint(len(sub_graph))]
            _visited_factors.add(root_factor)
            for edge in self._edges:
                if edge[0] == root_factor:
                    _to_visit.add(edge[1])
                elif edge[1] == root_factor:
                    _to_visit.add(edge[0])
        while len(_to_visit) > 0:
            next_factor = _to_visit.pop()
            next_edge = None
            for edge in self._edges:
                if edge[0] == next_factor and edge[1] in _visited_factors:
                    next_edge = (edge[1], edge[0])
                    _visited_factors.add(next_factor)
                elif edge[1] == next_factor and edge[0] in _visited_factors:
                    next_edge = edge
                    _visited_factors.add(next_factor)
                elif edge[0] == next_factor and edge[1] not in _visited_factors:
                    _to_visit.add(edge[1])
                elif edge[1] == next_factor and edge[0] not in _visited_factors:
                    _to_visit.add(edge[0])
            _forward_edges.append(next_edge)

        reversed_edges = [(edge[1], edge[0]) for edge in _forward_edges[::-1]]
        self._all_edges = _forward_edges + reversed_edges

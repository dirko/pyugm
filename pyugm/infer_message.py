"""
Module containing the inference routines.
"""
# License: BSD 3 clause

import numpy

from pyugm.factor import DiscreteFactor
from numba import jit, void, f8, i1, b1, njit, jit


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

    def _update_belief(self, edge, normalize=False, damping=0.0):
        """
        Helper to update the beliefs along a single edge.
        :param edge: A tuple of two factors.
        """
        old_separator = self._separator_potential[edge]
        variables_to_keep = old_separator.variable_set

        # Could probably be a bit faster
        new_separator = edge[0].marginalize(variables_to_keep)
        print 'updating ', edge
        print edge[0].data, edge[1].data
        print 'variables to keep', variables_to_keep
        print new_separator, new_separator.data
        new_separator_divided = edge[0].marginalize(variables_to_keep)
        print 'marginalized'
        print new_separator_divided, new_separator_divided.data
        multiply(new_separator_divided, old_separator, divide=True)
        print 'divided'
        print new_separator_divided, new_separator_divided.data
        multiply(edge[1], new_separator_divided, damping=damping)
        print 'multiplied'
        print new_separator_divided, new_separator_divided.data
        print edge[1], edge[1].data
        if normalize:
            edge[1].normalize()

        reverse_edge = (edge[1], edge[0])
        self._separator_potential[edge] = new_separator
        self._separator_potential[reverse_edge] = new_separator

        num_cells = numpy.prod(new_separator.data.shape) * 1.0
        average_change_per_cell = abs(new_separator.data - new_separator.rotate_other(old_separator)).sum() / num_cells

        return average_change_per_cell

    def calibrate(self, update_order=None, damping=0.0):
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
            average_change_per_cell = self._update_belief(edge, normalize=True, damping=damping)
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

        @jit
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
    def __init__(self, model, max_iterations=20, converge_delta=10**-10, callback=None):
        """
        Constructor.
        :param model: The model.
        :param max_iterations: The number of time to update each edge.
        :param converge_delta: If the potential table difference between the two iterations is less than this number,
                               updates will stop.
        """
        self._model = model
        self._edges = list(model.edges)
        self.current_iteration_delta = 0.0
        self.total_iterations = 0
        self._all_edges = []
        self._counter = 0
        self._max_iterations = max_iterations
        self._converge_delta = converge_delta
        self._callback = callback
        self.reset()

    def reset(self):
        """
        Reset the object.
        """
        self._counter = 0
        self._build_tree()

    def next_edge(self, last_update_change):
        """
        Get the next edge to update.
        :param last_update_change: Change in factor-potential that the previous update caused.
        :returns: An edge.
        """
        self.current_iteration_delta += last_update_change
        if self._counter >= len(self._all_edges):
            if self._callback:
                self._callback(self)
            if self.total_iterations > self._max_iterations or self.current_iteration_delta < self._converge_delta:
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
        numpy.random.shuffle(self._edges)
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


def multiply(this_factor, other_factor, divide=False, damping=0.0):
    """
    Multiply two factors.
    :param other_factor: The other factor to multiply into this factor.
    :param divide: If true then the other factor is divided into this factor, otherwise multiplied.
    """
    # pylint: disable=protected-access
    dim1 = len(other_factor.variables)
    dim2 = len(this_factor.variables)
    strides1 = numpy.array(other_factor._data.strides, dtype=numpy.int32) / other_factor._data.itemsize
    strides2 = numpy.array(this_factor._data.strides, dtype=numpy.int32) / this_factor._data.itemsize
    print 'strides', strides2, this_factor._data.strides, this_factor._data.itemsize, numpy.array(this_factor._data.strides, dtype=numpy.int32)
    card2 = numpy.array([this_factor.cardinalities[this_factor.axis_to_variable[axis]] for axis in xrange(dim2)],
                        dtype=numpy.int32)
    assignment1 = numpy.zeros(dim1, dtype=numpy.int32)
    assignment2 = numpy.zeros(dim2, dtype=numpy.int32)
    data1_flatshape = (numpy.prod(other_factor._data.shape),)
    data2_flatshape = (numpy.prod(this_factor._data.shape),)
    variable1_to_2 = numpy.array([this_factor.variable_to_axis[other_factor.axis_to_variable[ax1]]
                                  for ax1 in xrange(dim1)], dtype=numpy.int32)
    data1 = other_factor._data.view()
    data2 = this_factor._data.view()
    data1.shape = data1_flatshape
    data2.shape = data2_flatshape
    print 'pre mult'
    print other_factor.axis_to_variable, this_factor.axis_to_variable
    print variable1_to_2
    print other_factor._data.shape, this_factor._data.shape, this_factor._data.strides, this_factor._data.itemsize
    print strides2
    print
    _multiply_factors(data1, data2,
                     strides1, strides2,
                     card2,
                     assignment1, assignment2,
                     variable1_to_2, divide, damping)
    if divide:
        this_factor._log_normalizer -= other_factor._log_normalizer
    else:
        this_factor._log_normalizer += other_factor._log_normalizer
    # pylint: enable=protected-access


#@njit(void(f8[:], f8[:], i1[:], i1[:], i1[:], i1[:], i1[:], i1[:], b1, f8))
def _multiply_factors(data1, data2,
                     strides1, strides2,
                     cardinalities2,
                     assignment1, assignment2,
                     variable1_to_2, divide, damping):
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
    :param damping: Damping factor - float.
    """
    # pylint: disable=too-many-arguments
    # TODO: This is still quite slow - think about moving the complete update code to infer_message.py and to C or Cython.
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
        #print 'assignment', assignment1, assignment2
        # Get indices in data
        assignment1_index = 0
        for var1_i in range(len(strides1)):
            print ' + a', var1_i, assignment1_index, strides1[var1_i]
            assignment1_index += strides1[var1_i] * assignment1[var1_i]
        assignment2_index = 0
        for var2_i in range(len(strides2)):
            assignment2_index += strides2[var2_i] * assignment2[var2_i]
        # Multiply
        print 'assig index', assignment1, assignment2, assignment1_index, assignment2_index, data2[assignment2_index]
        if not divide:
            data2[assignment2_index] = ((1 - damping) * data1[assignment1_index] * data2[assignment2_index] +
                                        (damping * data2[assignment2_index]))
        else:
            if data2[assignment2_index] > 10e-300 and data1[assignment1_index] > 10e-300:
                data2[assignment2_index] = data2[assignment2_index] / data1[assignment1_index]

        # Tick variable2 assignment
        assignment2[0] += 1
        for var2_i in range(len(assignment2) - 1):
            if assignment2[var2_i] >= cardinalities2[var2_i]:
                assignment2[var2_i] = 0
                assignment2[var2_i + 1] += 1
        if assignment2[-1] >= cardinalities2[-1]:
            done = True

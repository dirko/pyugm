from factor import DiscreteFactor
from Queue import PriorityQueue, Queue, Empty
import numpy as np


class Model:
    def __init__(self, factor_list):
        self.factors = []  # factor objects
        self.cardinalities = dict()  # variable name to int
        self.variables_to_factors = dict()  # variable name to factor
        self.edges = set()  # pairs of factors
        self.disconnected_subgraphs = []  # list of sets of factors

        for factor in factor_list:
            self.add_factor(factor)

        self.parameters_to_index = {}
        # Get number of params and map param to position in param vector
        for factor in self.factors:
            if factor.parameters is not None:
                for parameter in factor.parameters.reshape(-1, ):
                    if isinstance(parameter, str):
                        self.parameters_to_index[parameter] = 0
        for index, key in enumerate(sorted(self.parameters_to_index.keys())):
            self.parameters_to_index[key] = index

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

                #print 'variable', variable, factors
                #print 'sepset', first_candidate_sepset
                #print 'marked', marked_factors, unmarked_factors
                #print
                while len(marked_factors) < len(factors):
                    largest_sepset = self.get_largest_unmarked_sepset(variable, list(factors), marked_factors,
                                                                      unmarked_factors)
                    # Add largest sepset factors to graph
                    if largest_sepset is not None:
                        add_edge_to_set(largest_sepset, self.edges)
                        mark_factors_in_edge(largest_sepset, marked_factors, unmarked_factors)

        self.find_disconnected_subgraphs()

    @staticmethod
    def get_largest_unmarked_sepset(variable, factors, marked_factors, unmarked_factors):
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

    def find_disconnected_subgraphs(self):
        """ Finds islands of factor nodes and adds them to disconnected_subgraphs. """

        def connected_factors(factor_to_follow):
            return_set = set()
            for edge in self.edges:
                if factor_to_follow in edge:
                    return_set.add(edge[0] if edge[1] == factor_to_follow else edge[1])
            return return_set

        visited_factors = set()
        for factor in self.factors:
            new_set = set()
            if factor not in visited_factors:
                new_set.add(factor)
                visited_factors.add(factor)
                factors_to_visit = connected_factors(factor)
                if len(factors_to_visit) > 0:
                    next_factor = factors_to_visit.pop()
                    while next_factor:
                        new_set.add(next_factor)
                        visited_factors.add(next_factor)
                        factors_to_visit = factors_to_visit.union(connected_factors(next_factor).difference(visited_factors))
                        try:
                            next_factor = factors_to_visit.pop()
                        except KeyError:
                            next_factor = None
                self.disconnected_subgraphs.append(new_set)

    def set_evidence(self, evidence, normalize=False):
        """ param evidence: list of (variable name, value) pairs """
        for variable, value in evidence.items():
            for factor in self.variables_to_factors[variable]:
                factor.set_evidence({variable: value}, normalize=normalize, inplace=True)

    def get_marginals(self, variable, normalize=True):
        """
        Return marginals of all the factors in which the variable appears.
        """
        return [factor.marginalize([variable], normalize=normalize) for factor in self.variables_to_factors[variable]]

    def set_parameters(self, parameters):
        """
        Iterate through factors and fill factor potentials with exp of these new parameters.
        """
        for factor in self.factors:
            original_shape = factor.data.shape
            new_data = factor.data.reshape(-1, )
            if factor.parameters is not None:
                for i, parameter in enumerate(factor.parameters.reshape(-1, )):
                    if isinstance(parameter, str):
                        new_data[i] = np.exp(parameters[self.parameters_to_index[parameter]])
                    else:
                        new_data[i] = parameter
                factor.data = new_data.reshape(original_shape)
                factor.log_normalizer = np.log(1.0)


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
        for sub_graph in model.disconnected_subgraphs:  # Roots
            root_factor = list(sub_graph)[0]
            self._visited_factors.add(root_factor)
            for edge in self._edges:
                if edge[0] == root_factor:
                    self._to_visit.add(edge[1])
                elif edge[1] == root_factor:
                    self._to_visit.add(edge[0])
        self._forward_edges = []
        self._direction = 'distribute'
        self.current_iteration_delta = 0.0
        self.total_iterations = 0

    def reset(self):
        self._to_visit = set()
        self._visited_factors = set()
        for sub_graph in self._model.disconnected_subgraphs:  # Roots
            root_factor = list(sub_graph)[0]
            self._visited_factors.add(root_factor)
            for edge in self._edges:
                if edge[0] == root_factor:
                    self._to_visit.add(edge[1])
                elif edge[1] == root_factor:
                    self._to_visit.add(edge[0])
        self._forward_edges = []
        self._direction = 'distribute'
        self.current_iteration_delta = 0.0
        self.total_iterations = 0

    def next_edge(self, last_update_change):
        """ Get the next edge to update """
        self.current_iteration_delta += last_update_change
        if self._direction == 'distribute':
            if len(self._to_visit) > 0:
                next_factor = self._to_visit.pop()
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
            else:
                self._direction = 'collect'
        if self._direction == 'collect':
            next_edge = None
            if len(self._forward_edges) > 0:
                next_edge_reversed = self._forward_edges.pop()
                next_edge = (next_edge_reversed[1], next_edge_reversed[0])
            elif self.total_iterations == 0:
                self.reset()
                self.total_iterations = 1
                next_edge = self.next_edge(last_update_change)
        return next_edge


class LoopyBeliefUpdateInference:
    def __init__(self, model):
        self.separator_potentials = dict()  # edge to pairs of separator potentials
        self.model = model

    def set_up_belief_update(self):
        """ Initialise separator potentials to 1 """
        for edge in list(self.model.edges):
            factor1, factor2 = edge
            sepset = factor1.variable_set.intersection(factor2.variable_set)
            separator_variables = [(variable, factor1.cardinalities[variable]) for variable in list(sepset)]
            # NOTE: will want to set this up more generically for other kinds of factors
            separator_factors = (DiscreteFactor(separator_variables), DiscreteFactor(separator_variables))
            self.separator_potentials[edge] = separator_factors
            self.separator_potentials[(edge[1], edge[0])] = separator_factors

    def update_belief(self, edge):
        old_separators = self.separator_potentials[edge]
        variables_to_keep = old_separators[0].variable_set

        # Could probably be a bit faster
        new_separator = edge[0].marginalize(variables_to_keep)
        new_separator_divided = edge[0].marginalize(variables_to_keep)
        new_separator_divided.multiply(old_separators[0], divide=True)
        edge[1].multiply(new_separator_divided)

        new_separators = (new_separator, old_separators[0])
        reverse_edge = (edge[1], edge[0])
        self.separator_potentials[edge] = new_separators
        self.separator_potentials[reverse_edge] = new_separators

        num_cells = np.prod(new_separator.data.shape) * 1.0
        average_change_per_cell = abs(new_separator.data - new_separator._rotate_other(old_separators[0])).sum() / num_cells

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
            total_z += np.log(np.sum(list(island)[0].data)) + list(island)[0].log_normalizer
        # Multiply so each factor has the same normaliser
        for island in self.model.disconnected_subgraphs:
            island_z = np.log(np.sum(list(island)[0].data)) + list(island)[0].log_normalizer
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
                new_factor.data[tuple(var[1] for var in instantiation)] *= potential_value
            done, instantiation = tick_instantiation(instantiation)

        return new_factor

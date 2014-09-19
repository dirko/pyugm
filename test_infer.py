import unittest
from numpy.testing import assert_array_almost_equal
import numpy as np

from factor import DiscreteFactor
from infer import Model, LoopyBeliefUpdateInference


def print_edge_set(edges):
    for edge in list(edges):
        print '({0}, {1})'.format(edge[0], edge[1])


def assertEdgeEqual(self, edge1, edge2, msg=''):
    if edge1 == edge2 or (edge1[1], edge1[0]) == edge2:
        pass
    else:
        raise AssertionError('Edges not equal, {0}) != {1}: \n{2}'.format(edge1, edge2, msg))


def assertEdgeSetsEqual(self, set1, set2, msg=''):
    set1_copy = set(list(set1))
    for edge in list(set1):
        set1_copy.add((edge[1], edge[0]))
    set2_copy = set(list(set2))
    for edge in list(set2):
        set2_copy.add((edge[1], edge[0]))
    try:
        self.assertSetEqual(set1_copy, set2_copy)
    except AssertionError:
        for edge in list(set1_copy):
            if edge not in set2_copy:
                print 'In 1 not in 2: ({0}, {1})'.format(edge[0], edge[1])
        for edge in list(set2_copy):
            if edge not in set1_copy:
                print 'In 2 not in 1: ({0}, {1})'.format(edge[0], edge[1])
        raise


class TestModel(unittest.TestCase):
    def test_get_largest_sepset_small(self):
        a = DiscreteFactor([(0, 2), (1, 2), (2, 2)])
        b = DiscreteFactor([(2, 2), (3, 2), (4, 2)])

        model = Model([a, b])
        model.build_graph()

        print model.edges
        print [(a, b)]
        assertEdgeEqual(self, list(model.edges)[0], (b, a))

    def test_get_largest_sepset_larger(self):
        a = DiscreteFactor([(0, 2), (1, 2), (2, 2)])
        b = DiscreteFactor([(0, 2), (3, 2), (4, 2)])
        c = DiscreteFactor([(1, 2), (2, 2), (5, 3), (6, 3)])

        model = Model([a, b, c])
        model.build_graph()

        # Expect:
        #  a{0 1 2} --[2 3]-- c{1 2 5 6}
        #      \
        #       [0]
        #          \
        #           b{0 3 4}
        print_edge_set(model.edges)
        assertEdgeSetsEqual(self, model.edges, {(a, c), (a, b)})

    def test_get_largest_sepset_large(self):
        a = DiscreteFactor([0, 1, 2, 3, 4, 5])
        b = DiscreteFactor([1, 2, 3, 4, 5, 6])
        c = DiscreteFactor([3, 4, 5, 6, 8])
        d = DiscreteFactor([0, 1, 2, 7])
        e = DiscreteFactor([1, 7, 8])

        # a{0 1 2 3 4 5} --[1 2 3 4 5]-- b{1 2 3 4 5 6} --[3 4 5 6]-- c{3 4 5 6 8}
        #      \                                                    /
        #    [0 1 2]                                            [8]
        #         \                                           /
        #       d{0 1 2 7} ---------[1 7]---------------   e{1 7 8}

        model = Model([a, b, c, d, e])
        model.build_graph()

        expected_edges = [(a, b), (b, c), (a, d), (d, e), (e, c)]

        print_edge_set(model.edges)
        assertEdgeSetsEqual(self, model.edges, expected_edges)

    def test_get_largest_sepset_grid(self):
        a = DiscreteFactor([0, 1])
        b = DiscreteFactor([1, 2, 3])
        c = DiscreteFactor([3, 4, 5])
        d = DiscreteFactor([5, 6])
        e = DiscreteFactor([0, 7, 8])
        f = DiscreteFactor([8, 2, 9, 10])
        g = DiscreteFactor([10, 4, 11, 12])
        h = DiscreteFactor([12, 13, 6])
        i = DiscreteFactor([7, 14])
        j = DiscreteFactor([14, 9, 15])
        k = DiscreteFactor([15, 11, 16])
        l = DiscreteFactor([16, 13])

        # a{0 1} ---[1]--- b{1 2 3} ---[3]--- c{3 4 5} ---[5]--- d{5 6}
        #   |                  |                   |                  |
        #  [0]                [2]                 [4]                [6]
        #   |                  |                   |                  |
        # e{0 7 8} --[8]-- f{8 2 9 10} --[10]- g{10 4 11 12} -[12]- h{12 13 6}
        #   |                  |                   |                  |
        #  [7]                [9]                 [11]               [13]
        #   |                  |                   |                  |
        # i{7 14} --[14]--j{14 9 15} --[15]-- k{15 11 16} --[16]-- l{16 13}

        model = Model([a, b, c, d, e, f, g, h, i, j, k, l])
        model.build_graph()

        expected_edges = {(a, b), (b, c), (c, d), (a, e), (b, f), (c, g), (d, h),
                          (e, f), (f, g), (g, h), (e, i), (f, j), (g, k), (h, l),
                          (i, j), (j, k), (k, l)}

        print_edge_set(model.edges)
        assertEdgeSetsEqual(self, model.edges, expected_edges)

    def test_set_evidence(self):
        a = DiscreteFactor([1, 2, 3], np.array(range(0, 8)).reshape((2, 2, 2)))
        b = DiscreteFactor([2, 3, 4], np.array(range(1, 9)).reshape((2, 2, 2)))
        model = Model([a, b])
        evidence = {2: 1, 4: 0}
        model.set_evidence(evidence)

        c = DiscreteFactor([1, 2, 3], np.array(range(0, 8)).reshape((2, 2, 2))).set_evidence(evidence)
        d = DiscreteFactor([2, 3, 4], np.array(range(1, 9)).reshape((2, 2, 2))).set_evidence(evidence)

        assert_array_almost_equal(c.data, model.factors[0].data)
        assert_array_almost_equal(d.data, model.factors[1].data)

    def test_set_parameters(self):
        a = DiscreteFactor([1, 2], parameters=np.array([[1, 2], ['a', 0.0]], dtype=object))
        b = DiscreteFactor([2, 3], parameters=np.array([['b', 'c'], ['d', 'a']]))
        model = Model([a, b])
        print a.parameters
        new_parameters = np.log(np.array([1, 2, 3, 4]))
        model.set_parameters(new_parameters)

        c = DiscreteFactor([1, 2], np.array([1, 2, 1, 0]).reshape((2, 2)))
        d = DiscreteFactor([2, 3], np.array([2, 3, 4, 1]).reshape((2, 2)))

        assert_array_almost_equal(c.data, model.factors[0].data)
        assert_array_almost_equal(d.data, model.factors[1].data)


class TestLoopyBeliefUpdateInference(unittest.TestCase):
    def test_set_up_separators(self):
        a = DiscreteFactor([(0, 2), (1, 2), (2, 2)])
        b = DiscreteFactor([(2, 2), (3, 2), (3, 2)])

        model = Model([a, b])
        model.build_graph()
        inference = LoopyBeliefUpdateInference(model)
        inference.set_up_belief_update()

        s = DiscreteFactor([(2, 2)])
        print inference.separator_potentials
        forward_edge = list(model.edges)[0]
        forward_and_backward_edge = [forward_edge, (forward_edge[1], forward_edge[0])]
        for edge in forward_and_backward_edge:
            separator_factor1 = inference.separator_potentials[edge][0]
            separator_factor2 = inference.separator_potentials[edge][1]

            self.assertSetEqual(separator_factor1.variable_set, s.variable_set)
            self.assertDictEqual(separator_factor1.cardinalities, s.cardinalities)
            assert_array_almost_equal(separator_factor1.data, s.data)

            self.assertSetEqual(separator_factor2.variable_set, s.variable_set)
            self.assertDictEqual(separator_factor2.cardinalities, s.cardinalities)
            assert_array_almost_equal(separator_factor2.data, s.data)

    def test_update_beliefs_small(self):
        a = DiscreteFactor([0, 1])
        b = DiscreteFactor([1, 2])
        model = Model([a, b])
        model.build_graph()
        inference = LoopyBeliefUpdateInference(model)
        inference.set_up_belief_update()
        #                       0
        #                     0  1
        # Phi* = Sum_{0} 1 0 [ 1 1 ]  =  1 0 [ 2 ]
        #                  1 [ 1 1 ]       1 [ 2 ]
        #
        #                                        1               1
        # Psi* = Phi* x Psi  =  1 0 [2] x 2 0 [ 1 1 ]  =  2 0 [ 2 2 ]
        #        Phi              1 [2]     1 [ 1 1 ]       1 [ 2 2 ]
        #
        #                        1           1
        # Phi** = Sum_{2} 2 0 [ 2 2 ]  =  [ 4 4 ]
        #                   1 [ 2 2 ]
        #
        #                            1              0               0
        # Psi** = Phi** x Psi  =  [ 2 2 ] x  1 0 [ 1 1 ]  =  1 0 [ 2 2 ]
        #         Phi*                         1 [ 1 1 ]       1 [ 2 2 ]
        #
        #             1
        # Phi*** = [ 4 4 ]
        #                                 1
        # Psi*** = Phi*** x Psi* = 2 0 [ 2 2 ]
        #          Phi**             1 [ 2 2 ]
        #
        change0, iterations0 = inference.update_beliefs(number_of_updates=2)
        change1, iterations1 = inference.update_beliefs(number_of_updates=3)
        print 'changes:', change0, change1, 'iterations:', iterations0, iterations1

        final_a = DiscreteFactor([0, 1])
        final_a.data *= 2
        final_b = DiscreteFactor([1, 2])
        final_b.data *= 2

        assert_array_almost_equal(a.get_data(), final_a.get_data())
        assert_array_almost_equal(b.get_data(), final_b.get_data())
        self.assertAlmostEqual(change1, 0, delta=10**-10)

    def test_update_beliefs_disconnected(self):
        a = DiscreteFactor([(1, 2), (2, 2)], data=np.array([[1, 2], [3, 4]]))
        b = DiscreteFactor([(2, 2), (3, 2)], data=np.array([[1, 2], [3, 4]]))
        c = DiscreteFactor([(4, 2), (5, 2)], data=np.array([[5, 6], [8, 9]]))
        d = DiscreteFactor([(5, 2), (6, 2)], data=np.array([[1, 6], [2, 3]]))
        e = DiscreteFactor([(7, 2), (8, 2)], data=np.array([[2, 1], [2, 3]]))

        model = Model([a, b, c, d, e])
        model.build_graph()
        for factor in model.factors:
            print 'before', factor, np.sum(factor.data)

        inference = LoopyBeliefUpdateInference(model)

        exhaustive_answer = inference.exhaustive_enumeration()  # do first because update_beliefs changes the factors
        print 'Exhaust', np.sum(exhaustive_answer.data)

        inference.set_up_belief_update()
        change = inference.update_beliefs(number_of_updates=35)
        print change

        for factor in model.factors:
            print factor, np.sum(factor.data)

        for factor in model.factors:
            self.assertAlmostEqual(np.sum(exhaustive_answer.get_data()), np.sum(factor.get_data()))
        self.assertAlmostEqual(exhaustive_answer.marginalize([7]).get_potential([(7, 1)]),
                               list(model.variables_to_factors[7])[0].marginalize([7]).get_potential([(7, 1)]))

    def test_belief_update_larger_tree(self):
        a = DiscreteFactor([0, 1], data=np.array([[1, 2], [2, 2]]))
        b = DiscreteFactor([1, 2], data=np.array([[3, 2], [1, 2]]))
        c = DiscreteFactor([2, 3], data=np.array([[1, 2], [3, 4]]))
        d = DiscreteFactor([3], data=np.array([2, 1]))
        e = DiscreteFactor([0], data=np.array([4, 1]))
        f = DiscreteFactor([2], data=np.array([1, 2]))
        #
        # a{0 1} - b{1 2} - c{2 3} - d{3}
        #    |       |
        # e{0}     f{2}
        #
        model = Model([a, b, c, d, e, f])
        model.build_graph()
        print 'edges', model.edges
        inference = LoopyBeliefUpdateInference(model)

        exhaustive_answer = inference.exhaustive_enumeration()  # do first because update_beliefs changes the factors

        print 'bp'
        inference.set_up_belief_update()
        change = inference.update_beliefs(number_of_updates=35)
        print change

        for factor in model.factors:
            print factor, np.sum(factor.data)

        self.assertAlmostEqual(np.sum(exhaustive_answer.get_data()), np.sum(a.get_data()))
        self.assertAlmostEqual(np.sum(exhaustive_answer.get_data()), np.sum(d.get_data()))

    def test_exhaustive_enumeration(self):
        a = DiscreteFactor([(0, 2), (1, 3)], data=np.array([[1, 2, 3], [4, 5, 6]]))
        b = DiscreteFactor([(0, 2), (2, 2)], data=np.array([[1, 2], [2, 1]]))
        # 0 1 2 |
        #-------+--------
        # 0 0 0 | 1x1=1
        # 0 0 1 | 1x2=2
        # 0 1 0 | 2x1=2
        # 0 1 1 | 2x2=4
        # 0 2 0 | 3x1=3
        # 0 2 1 | 3x2=6
        # 1 0 0 | 4x2=8
        # 1 0 1 | 4x1=4
        # 1 1 0 | 5x2=10
        # 1 1 1 | 5x1=5
        # 1 2 0 | 6x2=12
        # 1 2 1 | 6x1=6

        model = Model([a, b])
        model.build_graph()
        inference = LoopyBeliefUpdateInference(model)
        c = inference.exhaustive_enumeration()

        d = DiscreteFactor([(0, 2), (1, 3), (2, 2)])
        d.data = np.array([1, 2, 2, 4, 3, 6, 8, 4, 10, 5, 12, 6]).reshape(2, 3, 2)

        self.assertEqual(d.variables, c.variables)
        self.assertEqual(d.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(d.data, c.data)


if __name__ == '__main__':
    unittest.main()

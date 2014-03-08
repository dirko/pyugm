import unittest
from factor import Factor
from infer import Model
from numpy.testing import assert_array_almost_equal
import numpy as np


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


class TestBuildGraph(unittest.TestCase):
    def test_get_largest_sepset_small(self):
        a = Factor([(0, 2), (1, 2), (2, 2)])
        b = Factor([(2, 2), (3, 2), (4, 2)])

        model = Model([a, b])
        model.build_graph()

        print model.edges
        print [(a, b)]
        assertEdgeEqual(self, list(model.edges)[0], (b, a))

    def test_get_largest_sepset_larger(self):
        a = Factor([(0, 2), (1, 2), (2, 2)])
        b = Factor([(0, 2), (3, 2), (4, 2)])
        c = Factor([(1, 2), (2, 2), (5, 3), (6, 3)])

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
        a = Factor([0, 1, 2, 3, 4, 5])
        b = Factor([1, 2, 3, 4, 5, 6])
        c = Factor([3, 4, 5, 6, 8])
        d = Factor([0, 1, 2, 7])
        e = Factor([1, 7, 8])

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
        a = Factor([0, 1])
        b = Factor([1, 2, 3])
        c = Factor([3, 4, 5])
        d = Factor([5, 6])
        e = Factor([0, 7, 8])
        f = Factor([8, 2, 9, 10])
        g = Factor([10, 4, 11, 12])
        h = Factor([12, 13, 6])
        i = Factor([7, 14])
        j = Factor([14, 9, 15])
        k = Factor([15, 11, 16])
        l = Factor([16, 13])

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


class TestInference(unittest.TestCase):
    def test_set_up_separators(self):
        a = Factor([(0, 2), (1, 2), (2, 2)])
        b = Factor([(2, 2), (3, 2), (3, 2)])

        model = Model([a, b])
        model.build_graph()
        model.set_up_belief_update()

        s = Factor([(2, 2)])
        print model.separator_potentials
        forward_edge = list(model.edges)[0]
        forward_and_backward_edge = [forward_edge, (forward_edge[1], forward_edge[0])]
        for edge in forward_and_backward_edge:
            separator_factor1 = model.separator_potentials[edge][0]
            separator_factor2 = model.separator_potentials[edge][1]

            self.assertSetEqual(separator_factor1.variable_set, s.variable_set)
            self.assertDictEqual(separator_factor1.cardinalities, s.cardinalities)
            assert_array_almost_equal(separator_factor1.data, s.data)

            self.assertSetEqual(separator_factor2.variable_set, s.variable_set)
            self.assertDictEqual(separator_factor2.cardinalities, s.cardinalities)
            assert_array_almost_equal(separator_factor2.data, s.data)

    def test_set_up_update_queue(self):
        a = Factor([(0, 2), (1, 2), (2, 2)])
        b = Factor([(2, 2), (3, 3), (4, 2)])
        c = Factor([(3, 3), (4, 2), (5, 2)])

        model = Model([a, b, c])
        model.build_graph()
        model.set_up_belief_update()

        priority_edges = set()
        while not model.belief_update_queue.empty():
            edge = model.belief_update_queue.get_nowait()
            priority_edges.add(edge)

        print priority_edges
        print model.belief_update_queue.qsize()
        expected_set = {(a, b), (b, c)}
        priority_edges_set = set()
        for priority_edge in priority_edges:
            self.assertEqual(priority_edge[0], -np.inf)
            priority_edges_set.add(priority_edge[1])
        assertEdgeSetsEqual(self, priority_edges_set, expected_set)

    def test_update_beliefs_small(self):
        a = Factor([0, 1])
        b = Factor([1, 2])
        model = Model([a, b])
        model.build_graph()
        model.set_up_belief_update()
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
        change0 = model.update_beliefs(number_of_updates=2)
        change1 = model.update_beliefs(number_of_updates=3)
        print change0, change1

        final_a = Factor([0, 1])
        final_a.data *= 2
        final_b = Factor([1, 2])
        final_b.data *= 2

        assert_array_almost_equal(a.data, final_a.data)
        assert_array_almost_equal(b.data, final_b.data)
        self.assertAlmostEqual(change1, 0, delta=10**-10)

    def test_belief_update_larger_tree(self):
        a = Factor([0, 1], data=np.array([[1, 2], [2, 2]]))
        b = Factor([1, 2], data=np.array([[3, 2], [1, 2]]))
        c = Factor([2, 3], data=np.array([[1, 2], [3, 4]]))
        d = Factor([3], data=np.array([2, 1]))
        e = Factor([0], data=np.array([4, 1]))
        f = Factor([2], data=np.array([1, 2]))
        #
        # a{0 1} - b{1 2} - c{2 3} - d{3}
        #    |       |
        # e{0}     f{2}
        #
        model = Model([a, b, c, d, e, f])
        model.build_graph()

        exhaustive_answer = model.exhaustive_enumeration()  # do first because update_beliefs changes the factors

        model.set_up_belief_update()
        for epoch in xrange(5):
            change = model.update_beliefs(number_of_updates=20)
            print epoch, '==============================', change, '==============================='
        for factor in model.factors:
            print factor, np.sum(factor.data)

        self.assertAlmostEqual(np.sum(exhaustive_answer.data), np.sum(a.data))
        self.assertAlmostEqual(np.sum(exhaustive_answer.data), np.sum(d.data))

    def test_exhaustive_enumeration(self):
        a = Factor([(0, 2), (1, 3)], data=np.array([[1, 2, 3], [4, 5, 6]]))
        b = Factor([(0, 2), (2, 2)], data=np.array([[1, 2], [2, 1]]))
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
        c = model.exhaustive_enumeration()

        d = Factor([(0, 2), (1, 3), (2, 2)])
        d.data = np.array([1, 2, 2, 4, 3, 6, 8, 4, 10, 5, 12, 6]).reshape(2, 3, 2)

        self.assertEqual(d.variables, c.variables)
        self.assertEqual(d.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(d.data, c.data)

if __name__ == '__main__':
    unittest.main()

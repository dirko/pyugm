import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np

from pyugm.factor import DiscreteFactor
from pyugm.infer import LoopyBeliefUpdateInference
from pyugm.infer import FloodingProtocol
from pyugm.infer import DistributeCollectProtocol
from pyugm.infer import ExhaustiveEnumeration
from pyugm.model import Model


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


class TestLoopyBeliefUpdateInference(unittest.TestCase):
    def test_set_up_separators(self):
        a = DiscreteFactor([(0, 2), (1, 2), (2, 2)])
        b = DiscreteFactor([(2, 2), (3, 2), (3, 2)])

        model = Model([a, b])
        inference = LoopyBeliefUpdateInference(model)

        s = DiscreteFactor([(2, 2)])
        print inference._separator_potential
        forward_edge = list(model.edges)[0]
        forward_and_backward_edge = [forward_edge, (forward_edge[1], forward_edge[0])]
        for edge in forward_and_backward_edge:
            separator_factor = inference._separator_potential[edge]

            self.assertSetEqual(separator_factor.variable_set, s.variable_set)
            self.assertDictEqual(separator_factor.cardinalities, s.cardinalities)
            assert_array_almost_equal(separator_factor.data, s._data)

    def test_update_beliefs_small(self):
        a = DiscreteFactor([0, 1])
        b = DiscreteFactor([1, 2])
        model = Model([a, b])
        inference = LoopyBeliefUpdateInference(model)
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
        update_order1 = FloodingProtocol(model=model, max_iterations=2)
        change0, iterations0 = inference.calibrate(update_order1)
        update_order2 = FloodingProtocol(model=model, max_iterations=3)
        change1, iterations1 = inference.calibrate(update_order2)
        print 'changes:', change0, change1, 'iterations:', iterations0, iterations1

        final_a = DiscreteFactor([0, 1])
        final_a._data *= 2
        final_b = DiscreteFactor([1, 2])
        final_b._data *= 2

        assert_array_almost_equal(a.data, final_a.data)
        assert_array_almost_equal(b.data, final_b.data)
        self.assertAlmostEqual(change1, 0, delta=10**-10)

    def test_update_beliefs_disconnected(self):
        a = DiscreteFactor([(1, 2), (2, 2)], data=np.array([[1, 2], [3, 4]], dtype=np.float64))
        b = DiscreteFactor([(2, 2), (3, 2)], data=np.array([[1, 2], [3, 4]], dtype=np.float64))
        c = DiscreteFactor([(4, 2), (5, 2)], data=np.array([[5, 6], [8, 9]], dtype=np.float64))
        d = DiscreteFactor([(5, 2), (6, 2)], data=np.array([[1, 6], [2, 3]], dtype=np.float64))
        e = DiscreteFactor([(7, 2), (8, 2)], data=np.array([[2, 1], [2, 3]], dtype=np.float64))

        model = Model([a, b, c, d, e])
        for factor in model.factors:
            print 'before', factor, np.sum(factor.data)

        inference = LoopyBeliefUpdateInference(model)

        exact_inference = ExhaustiveEnumeration(model)
        # do first because update_beliefs changes the factors
        exhaustive_answer = exact_inference.exhaustively_enumerate()
        print 'Exhaust', np.sum(exhaustive_answer.data)

        update_order = DistributeCollectProtocol(model)
        change = inference.calibrate(update_order=update_order)
        print change

        for factor in model.factors:
            print factor, np.sum(factor.data)

        for factor in model.factors:
            self.assertAlmostEqual(np.sum(exhaustive_answer.data), np.sum(factor.data))
        self.assertAlmostEqual(exhaustive_answer.marginalize([7]).get_potential([(7, 1)]),
                               list(model._variables_to_factors[7])[0].marginalize([7]).get_potential([(7, 1)]))

    def test_belief_update_larger_tree(self):
        a = DiscreteFactor([0, 1], data=np.array([[1, 2], [2, 2]], dtype=np.float64))
        b = DiscreteFactor([1, 2], data=np.array([[3, 2], [1, 2]], dtype=np.float64))
        c = DiscreteFactor([2, 3], data=np.array([[1, 2], [3, 4]], dtype=np.float64))
        d = DiscreteFactor([3], data=np.array([2, 1], dtype=np.float64))
        e = DiscreteFactor([0], data=np.array([4, 1], dtype=np.float64))
        f = DiscreteFactor([2], data=np.array([1, 2], dtype=np.float64))
        #
        # a{0 1} - b{1 2} - c{2 3} - d{3}
        #    |       |
        # e{0}     f{2}
        #
        model = Model([a, b, c, d, e, f])
        print 'edges', model.edges
        inference = LoopyBeliefUpdateInference(model)

        exact_inference = ExhaustiveEnumeration(model)
        # do first because update_beliefs changes the factors
        exhaustive_answer = exact_inference.exhaustively_enumerate()

        print 'bp'
        update_order = DistributeCollectProtocol(model)
        change = inference.calibrate(update_order=update_order)
        print change

        for factor in model.factors:
            print factor, np.sum(factor.data) + np.exp(factor.log_normalizer)

        self.assertAlmostEqual(np.sum(exhaustive_answer.data), np.sum(a.data))
        self.assertAlmostEqual(np.sum(exhaustive_answer.data), np.sum(d.data))

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
        exact_inference = ExhaustiveEnumeration(model)
        c = exact_inference.exhaustively_enumerate()

        d = DiscreteFactor([(0, 2), (1, 3), (2, 2)])
        d._data = np.array([1, 2, 2, 4, 3, 6, 8, 4, 10, 5, 12, 6]).reshape(2, 3, 2)

        self.assertEqual(d.variables, c.variables)
        self.assertEqual(d.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(d._data, c.data)


if __name__ == '__main__':
    unittest.main()

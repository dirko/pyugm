"""
Tests the inference module.
"""

# pylint: disable=protected-access
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-many-public-methods

import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np

from pyugm.factor import DiscreteFactor
from pyugm.infer_message import LoopyBeliefUpdateInference
from pyugm.infer_message import TreeBeliefUpdateInference
from pyugm.infer_message import FloodingProtocol
from pyugm.infer_message import DistributeCollectProtocol
from pyugm.infer_message import LoopyDistributeCollectProtocol
from pyugm.infer_message import multiply
from pyugm.infer_message import ExhaustiveEnumeration
from pyugm.model import Model
from pyugm.tests.test_utils import GraphTestCase


class TestFactorMultiplication(unittest.TestCase):
    def test_multiply_small_inplace(self):
        a = DiscreteFactor([(0, 2), (1, 2)])
        a._data[0, 0] = 1
        a._data[0, 1] = 2
        a._data[1, 0] = 5
        a._data[1, 1] = 6
        a._log_normalizer = 0.0

        b = DiscreteFactor([(1, 2)])
        b._data[0] = 2
        b._data[1] = 3
        b._log_normalizer = 0.0

        c = DiscreteFactor([(0, 2), (1, 2)])
        c._data[0, 0] = 2
        c._data[0, 1] = 6
        c._data[1, 0] = 10
        c._data[1, 1] = 18
        c._log_normalizer = 0.0

        multiply(a, b)

        print a._data
        print a._log_normalizer
        print c._data
        print c._log_normalizer
        print a._data.shape
        print c._data.shape
        self.assertEqual(a.variables, c.variables)
        self.assertEqual(a.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(a.data, c.data)

    def test_multiply_small_a(self):
        a = DiscreteFactor([(0, 2), (1, 2)])
        a._data[0, 0] = 1
        a._data[0, 1] = 2
        a._data[1, 0] = 5
        a._data[1, 1] = 6
        a._log_normalizer = 0.0

        e = DiscreteFactor([(0, 2)])
        e._data[0] = 2
        e._data[1] = 3
        e._log_normalizer = 0.0

        f = DiscreteFactor([(0, 2), (1, 2)])
        f._data[0, 0] = 1 * 2
        f._data[0, 1] = 2 * 2
        f._data[1, 0] = 5 * 3
        f._data[1, 1] = 6 * 3
        f._log_normalizer = 0.0

        multiply(a, e)

        print 'a', a._data
        print a._log_normalizer
        print 'e', e._data
        print e._log_normalizer
        print
        print a._data
        print a._log_normalizer
        print f._data
        print f._log_normalizer
        print a._data.shape
        print f._data.shape
        self.assertEqual(a.variables, f.variables)
        self.assertEqual(a.axis_to_variable, f.axis_to_variable)
        assert_array_almost_equal(a.data, f.data)

    def test_multiply_larger(self):
        a = DiscreteFactor([(0, 2), (3, 2), (12, 3)])
        a._data[0, 0, 0] = 2
        a._data[0, 0, 1] = 1
        a._data[0, 0, 2] = 2
        a._data[0, 1, 0] = 3
        a._data[0, 1, 1] = 7
        a._data[0, 1, 2] = 4

        a._data[1, 0, 0] = 1
        a._data[1, 0, 1] = 1
        a._data[1, 0, 2] = 3
        a._data[1, 1, 0] = 4
        a._data[1, 1, 1] = 9
        a._data[1, 1, 2] = 10

        b = DiscreteFactor([(0, 2), (12, 3)])
        b._data[0, 0] = 2
        b._data[0, 1] = 3
        b._data[0, 2] = 1
        b._data[1, 0] = 5
        b._data[1, 1] = 1
        b._data[1, 2] = 7

        c = DiscreteFactor([(0, 2), (3, 2), (12, 3)])
        c._data[0, 0, 0] = 2 * 2
        c._data[0, 0, 1] = 1 * 3
        c._data[0, 0, 2] = 2 * 1
        c._data[0, 1, 0] = 3 * 2
        c._data[0, 1, 1] = 7 * 3
        c._data[0, 1, 2] = 4 * 1

        c._data[1, 0, 0] = 1 * 5
        c._data[1, 0, 1] = 1 * 1
        c._data[1, 0, 2] = 3 * 7
        c._data[1, 1, 0] = 4 * 5
        c._data[1, 1, 1] = 9 * 1
        c._data[1, 1, 2] = 10 * 7

        multiply(a, b)

        print a._data
        print c._data
        print a._data.shape
        print c._data.shape
        self.assertEqual(a.variables, c.variables)
        self.assertEqual(a.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(a.data, c.data)

    def test_multiply_larger_correct_order(self):
        a = DiscreteFactor([(0, 2), (3, 2), (12, 3)])
        a._data[0, 0, 0] = 2
        a._data[0, 0, 1] = 1
        a._data[0, 0, 2] = 2
        a._data[0, 1, 0] = 3
        a._data[0, 1, 1] = 7
        a._data[0, 1, 2] = 4

        a._data[1, 0, 0] = 1
        a._data[1, 0, 1] = 1
        a._data[1, 0, 2] = 3
        a._data[1, 1, 0] = 4
        a._data[1, 1, 1] = 9
        a._data[1, 1, 2] = 10

        b = DiscreteFactor([(12, 3), (0, 2)])
        b._data[0, 0] = 2
        b._data[1, 0] = 3
        b._data[2, 0] = 1
        b._data[0, 1] = 5
        b._data[1, 1] = 1
        b._data[2, 1] = 7

        c = DiscreteFactor([(0, 2), (3, 2), (12, 3)])
        c._data[0, 0, 0] = 2 * 2
        c._data[0, 0, 1] = 1 * 3
        c._data[0, 0, 2] = 2 * 1
        c._data[0, 1, 0] = 3 * 2
        c._data[0, 1, 1] = 7 * 3
        c._data[0, 1, 2] = 4 * 1

        c._data[1, 0, 0] = 1 * 5
        c._data[1, 0, 1] = 1 * 1
        c._data[1, 0, 2] = 3 * 7
        c._data[1, 1, 0] = 4 * 5
        c._data[1, 1, 1] = 9 * 1
        c._data[1, 1, 2] = 10 * 7

        multiply(a, b)

        print a._data
        print c._data
        print a._data.shape
        print c._data.shape
        self.assertEqual(a.variables, c.variables)
        self.assertEqual(a.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(a.data, c.data)

    def test_divide_small(self):
        a = DiscreteFactor([(0, 2), (1, 2)], data=np.array([[1.0, 2], [5, 6]]))
        print 'a', a._log_normalizer
        #a.data[0, 0] = 1.0
        #a.data[0, 1] = 2.0
        #a.data[1, 0] = 5.0
        #a.data[1, 1] = 6.0

        b = DiscreteFactor([(1, 2)], data=np.array([2.0, 3]))
        print 'b', b._log_normalizer
        #b.data[0] = 2.0
        #b.data[1] = 3.0

        data = np.array([[1.0 / 2.0, 2.0 / 3.0],
                         [5.0 / 2.0, 6.0 / 3.0]])
        c = DiscreteFactor([(0, 2), (1, 2)], data=data)
        print 'c', c._log_normalizer
        #c.data[0, 0] = 1.0 / 2.0
        #c.data[0, 1] = 2.0 / 3.0
        #c.data[1, 0] = 5.0 / 2.0
        #c.data[1, 1] = 6.0 / 3.0

        multiply(a, b, divide=True)

        print a._data
        print a._log_normalizer
        print c._data
        print c._log_normalizer
        print a._data.shape
        print c._data.shape
        self.assertEqual(a.variables, c.variables)
        self.assertEqual(a.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(a.data, c.data)


class TestTreeBeliefUpdateInference(GraphTestCase):
    def test_set_up_separators(self):
        a = DiscreteFactor([(0, 2), (1, 2), (2, 2)])
        b = DiscreteFactor([(2, 2), (3, 2), (3, 2)])

        model = Model([a, b])
        inference = TreeBeliefUpdateInference(model)

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
        inference = TreeBeliefUpdateInference(model)
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

        inference = TreeBeliefUpdateInference(model)

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
        inference = TreeBeliefUpdateInference(model)

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

    def test_belief_update_long_tree(self):
        #pair_template = np.array([['alpha', 'beta'], ['gamma', 'delta']])
        label_template = np.array([['same', 'different'],
                                   ['different', 'same']])
        #obs_template = np.array([['oalpha', 'obeta'], ['ogamma', 'odelta']])
        #observation_template = np.array([['obs_low', 'obs_high'],
        #                                 ['obs_high', 'obs_low']])
        observation_template = np.array([['obs_low'] * 32,
                                         ['obs_high'] * 32])
        observation_template[0, 13:17] = 'obs_high'
        observation_template[1, 13:17] = 'obs_low'
        N = 2
        pairs = [DiscreteFactor([(i, 2), (i + 1, 2)], parameters=label_template) for i in xrange(N - 1)]
        obs = [DiscreteFactor([(i, 2), (i + N, 32)], parameters=observation_template) for i in xrange(N)]
        repe = [ 16.,  16.,  14.,  13.,  15.,  16.,  14.,  13.,  15.,  16.,  15.,
        13.,  14.,  16.,  16.,  15.,  13.,  13.,  14.,  14.,  13.,  14.,
        14.,  14.,  14.,  14.,  14.,  14.,  14.,  14.,  14.,  14.,  14.,
        14.,  14.,  14.,  14.,  14.,  14.,  14.,  14.,   9.,   4.,   4.,
         4.,   4.,   5.,   3.,   2.,   3.,   2.,   3.,   3.,   3.,   3.,
         3.,   3.,   3.,   3.,   4.,   4.,   5.,   5.,   5.]
        #evidence = dict((i + N, 0 if repe[i % len(repe)] >= 13 and repe[i % len(repe)] < 17 else 1) for i in xrange(N))
        evidence = dict((i + N, 0) for i in xrange(N))

        model = Model(pairs + obs)
        parameters = {'same': 2.0, 'different': -1.0, 'obs_high': 0.0, 'obs_low': -0.0}
        model.set_parameters(parameters=parameters)
        model.set_evidence(evidence)

        #inference = TreeBeliefUpdateInference(model)
        inference = LoopyBeliefUpdateInference(model)

        print 'enumerating'

        for i in xrange(N):
            #expected_marginal = exhaustive_answer.marginalize([i])
            expected_marginal = model.get_marginals(i)[0]
            for actual_marginal in model.get_marginals(i):
                print i, evidence[i + N], expected_marginal.normalized_data, actual_marginal.normalized_data, \
                    sum(abs(expected_marginal.normalized_data - actual_marginal.normalized_data))
        print '-' * 20
        for factor in model.factors:
            print factor, factor.data
        print '-' * 10
        exact_inference = ExhaustiveEnumeration(model)
        # do first because update_beliefs changes the factors
        exhaustive_answer = exact_inference.exhaustively_enumerate()
        def reporter(ordering):
            change = ordering.current_iteration_delta
            print ordering.total_iterations, change

        print 'bp'
        #update_order = DistributeCollectProtocol(model)
        update_order = FloodingProtocol(model, max_iterations=4, callback=reporter)
        change = inference.calibrate(update_order=update_order)
        print 'change', change

        #print np.sum(exhaustive_answer.data), np.sum(pairs[0].data)
        #print exhaustive_answer.marginalize([4]).data, model.get_marginals(4)[0].data
        for i in xrange(N):
            expected_marginal = exhaustive_answer.marginalize([i])
            #expected_marginal = model.get_marginals(i)[0]
            for actual_marginal in model.get_marginals(i):
                print i, evidence[i + N], expected_marginal.normalized_data, actual_marginal.normalized_data,\
                    sum(abs(expected_marginal.normalized_data - actual_marginal.normalized_data))
        print '-' * 20
        for factor in model.factors:
            print factor, factor.data


        for i in xrange(N):
            expected_marginal = exhaustive_answer.marginalize([i])
            #expected_marginal = model.get_marginals(i)[0]
            for actual_marginal in model.get_marginals(i):
                assert_array_almost_equal(expected_marginal.normalized_data, actual_marginal.normalized_data)
        #self.assertAlmostEqual(np.sum(exhaustive_answer.data), np.sum(pairs[0].data))
        #self.assertAlmostEqual(np.sum(exhaustive_answer.data), np.sum(pairs[-1].data))


class TestLoopyBeliefUpdateInference(GraphTestCase):
    def test_loopy_distribute_collect(self):
        a = DiscreteFactor([0, 1], data=np.array([[1, 2], [2, 2]], dtype=np.float64))
        b = DiscreteFactor([1, 2], data=np.array([[3, 2], [1, 2]], dtype=np.float64))
        c = DiscreteFactor([2, 0], data=np.array([[1, 2], [3, 4]], dtype=np.float64))
        #
        # a{0 1} - b{1 2}
        #    \       /
        #      c{2 0}
        #
        # a{0 1} - {0} - c{2 0}
        #
        #
        #
        #
        model = Model([a, b, c])
        inference = LoopyBeliefUpdateInference(model)

        exact_inference = ExhaustiveEnumeration(model)
        exhaustive_answer = exact_inference.exhaustively_enumerate()

        update_order = LoopyDistributeCollectProtocol(model, max_iterations=40)
        change = inference.calibrate(update_order=update_order)
        print change

        for factor in model.factors:
            print factor, np.sum(factor.data), factor.log_normalizer
        for var in model._variables_to_factors.keys():
            print var, exhaustive_answer.marginalize([var]).data, exhaustive_answer.marginalize([var]).log_normalizer
        print
        for var in model._variables_to_factors.keys():
            print var, model.get_marginals(var)[0].data, model.get_marginals(var)[0].log_normalizer

        for variable in model.variables:
            for factor in model.get_marginals(variable):
                expected_table = exhaustive_answer.marginalize([variable])
                actual_table = factor.marginalize([variable])
                assert_array_almost_equal(expected_table.normalized_data, actual_table.normalized_data, decimal=2)

    def test_loopy_distribute_collect_grid(self):
        a = DiscreteFactor([0, 1], data=np.random.randn(2, 2))
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

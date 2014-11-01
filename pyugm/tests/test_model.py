"""
Tests for the module.
"""
# pylint: disable=protected-access
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-many-public-methods

import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np

from pyugm.factor import DiscreteFactor
from pyugm.model import Model
from pyugm.tests.test_utils import GraphTestCase
from pyugm.tests.test_utils import print_edge_set


class TestModel(GraphTestCase):
    def test_get_largest_sepset_small(self):
        a = DiscreteFactor([(0, 2), (1, 2), (2, 2)])
        b = DiscreteFactor([(2, 2), (3, 2), (4, 2)])

        model = Model([a, b])

        print model.edges
        print [(a, b)]
        self._assert_edge_equal(list(model.edges)[0], (b, a))

    def test_get_largest_sepset_larger(self):
        a = DiscreteFactor([(0, 2), (1, 2), (2, 2)])
        b = DiscreteFactor([(0, 2), (3, 2), (4, 2)])
        c = DiscreteFactor([(1, 2), (2, 2), (5, 3), (6, 3)])

        model = Model([a, b, c])

        # Expect:
        #  a{0 1 2} --[2 3]-- c{1 2 5 6}
        #      \
        #       [0]
        #          \
        #           b{0 3 4}
        print_edge_set(model.edges)
        self._assert_edge_sets_equal(model.edges, {(a, c), (a, b)})

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

        expected_edges = [(a, b), (b, c), (a, d), (d, e), (e, c)]

        print_edge_set(model.edges)
        self._assert_edge_sets_equal(model.edges, expected_edges)

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

        expected_edges = {(a, b), (b, c), (c, d), (a, e), (b, f), (c, g), (d, h),
                          (e, f), (f, g), (g, h), (e, i), (f, j), (g, k), (h, l),
                          (i, j), (j, k), (k, l)}

        print_edge_set(model.edges)
        self._assert_edge_sets_equal(model.edges, expected_edges)

    @staticmethod
    def test_set_evidence():
        a = DiscreteFactor([1, 2, 3], np.array(range(0, 8)).reshape((2, 2, 2)))
        b = DiscreteFactor([2, 3, 4], np.array(range(1, 9)).reshape((2, 2, 2)))
        model = Model([a, b])
        evidence = {2: 1, 4: 0}
        model.set_evidence(evidence)

        c = DiscreteFactor([1, 2, 3], np.array(range(0, 8)).reshape((2, 2, 2)))
        c.set_evidence(evidence)
        d = DiscreteFactor([2, 3, 4], np.array(range(1, 9)).reshape((2, 2, 2)))
        d.set_evidence(evidence)

        assert_array_almost_equal(c._data, model.factors[0].data)
        assert_array_almost_equal(d._data, model.factors[1].data)

    @staticmethod
    def test_set_parameters():
        a = DiscreteFactor([1, 2], parameters=np.array([[1, 2], ['a', 0.0]], dtype=object))
        b = DiscreteFactor([2, 3], parameters=np.array([['b', 'c'], ['d', 'a']]))
        model = Model([a, b])
        print a.parameters
        new_parameters = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        model.set_parameters(new_parameters)

        c = DiscreteFactor([1, 2], np.array([1, 2, np.exp(1), 0]).reshape((2, 2)))
        d = DiscreteFactor([2, 3], np.array([np.exp(2), np.exp(3), np.exp(4), np.exp(1)]).reshape((2, 2)))

        assert_array_almost_equal(c._data, model.factors[0].data)
        assert_array_almost_equal(d._data, model.factors[1].data)


if __name__ == '__main__':
    unittest.main()

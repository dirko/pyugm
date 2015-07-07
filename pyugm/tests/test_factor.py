"""
Tests for the factor module.
"""
# pylint: disable=protected-access
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-many-public-methods

import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np

from pyugm.factor import DiscreteFactor, DiscreteBelief


class TestFactor(unittest.TestCase):
    """
    Tests for the factor class.
    """
    def test_marginalize_small_edge(self):
        data = np.array([[0, 2],
                         [5, 8]])
        a = DiscreteFactor([(0, 2), (1, 2)], data=data)

        print a.data
        b = a.marginalize([0, 1])
        print
        print b.data
        print b.data.shape
        self.assertEqual(b.variables, a.variables)
        self.assertEqual(b.axis_to_variable, a.axis_to_variable)
        assert_array_almost_equal(b.data, a.data)

        c = a.marginalize([1, 0])
        print c.data
        print c.data.shape
        self.assertEqual(c.variables, a.variables)
        self.assertEqual(c.axis_to_variable, a.axis_to_variable)
        assert_array_almost_equal(c.data, a.data)

    def test_marginalize_small(self):
        data = np.array([[1, 2],
                         [5, 8]])
        a = DiscreteFactor([(0, 2), (1, 2)], data=data)

        data = np.array([3, 13])
        c = DiscreteFactor([(0, 2)], data=data)

        b = a.marginalize([0])
        print b.data
        print c.data
        print b.data.shape
        print c.data.shape
        self.assertEqual(b.variables, c.variables)
        self.assertEqual(b.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(b.data, c.data)

        data = np.array([6, 10])
        e = DiscreteFactor([(1, 2)], data)

        d = a.marginalize([1])
        print d.data
        print e.data
        print d.data.shape
        print e.data.shape

        self.assertEqual(d.variables, e.variables)
        self.assertEqual(d.axis_to_variable, e.axis_to_variable)
        assert_array_almost_equal(d.data, e.data)

    def test_marginalize_larger(self):
        data = np.array([[[1, 2],
                          [5, 8],
                          [9, 10]],
                        [[11, 12],
                         [15, 18],
                         [19, 21]]])
        a = DiscreteFactor([(0, 2), (4, 3), (20, 2)], data=data)

        data = np.array([35, 96])
        c = DiscreteFactor([(0, 2)], data=data)

        b = a.marginalize([0])
        print b.data
        print c.data
        print b.data.shape
        print c.data.shape
        self.assertEqual(b.variables, c.variables)
        self.assertEqual(b.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(b.data, c.data)

        data = np.array([[12, 14],
                         [20, 26],
                         [28, 31]])
        e = DiscreteFactor([(4, 3), (20, 2)], data=data)

        d = a.marginalize([4, 20])
        print d.data
        print e.data
        print d.data.shape
        print e.data.shape
        self.assertEqual(d.variables, e.variables)
        self.assertEqual(d.axis_to_variable, e.axis_to_variable)
        assert_array_almost_equal(d.data, e.data)

    def test_get_potential_single(self):
        a = DiscreteFactor([(4, 2), (8, 3)], data=np.array(range(6)).reshape(2, 3))
        b = a.get_potential([(8, 0), (4, 1), (2, 4)])
        print b
        self.assertAlmostEqual(b, 3)

    def test_get_potential_slice(self):
        a = DiscreteFactor([(4, 2), (8, 3)], data=np.array(range(6)).reshape(2, 3))
        b = a.get_potential([(8, 0), (9, 1), (2, 4)])
        self.assertIsNone(assert_array_almost_equal(b, np.array([0, 3])))


class TestBelief(unittest.TestCase):
    def test_set_evidence_not_normalized_inplace(self):
        af = DiscreteFactor([(1, 2), (4, 3)], data=np.array(range(6)).reshape(2, 3))
        a = DiscreteBelief(af)
        print a._data
        a.set_evidence({1: 1})
        c_data = np.array([[0, 0, 0], [3, 4, 5]])
        c = DiscreteFactor([(1, 2), (4, 3)], data=c_data)
        self.assertItemsEqual(c.variables, a.variables)
        assert_array_almost_equal(c.data, a.data)

if __name__ == '__main__':
    unittest.main()

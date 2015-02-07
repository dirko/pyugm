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

from pyugm.factor import DiscreteFactor


class TestFactor(unittest.TestCase):
    """
    Tests for the factor class.
    """
    def test_marginalize_small_edge(self):
        a = DiscreteFactor([(0, 2), (1, 2)])
        # a.data[0, 0] should equal 1 by default
        a._data[0, 1] = 2
        a._data[1, 0] = 5
        a._data[1, 1] = 8

        print a.data
        print a._log_normalizer
        b = a.marginalize([0, 1])
        print
        print b.data
        print b.log_normalizer
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
        a = DiscreteFactor([(0, 2), (1, 2)])
        a._data[0, 0] = 1
        a._data[0, 1] = 2
        a._data[1, 0] = 5
        a._data[1, 1] = 8

        c = DiscreteFactor([(0, 2)])
        c._data[0] = 3
        c._data[1] = 13

        b = a.marginalize([0])
        print b.data
        print b.log_normalizer
        print c._data
        print b.data.shape
        print c._data.shape
        #self.assertEqual(b.data[0], c.data[0])
        #self.assertEqual(b.data[1], c.data[1])
        #self.assertEqual(b.data, c.data)
        self.assertEqual(b.variables, c.variables)
        self.assertEqual(b.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(b.data, c.data)

        e = DiscreteFactor([(1, 2)])
        e._data[0] = 6
        e._data[1] = 10

        d = a.marginalize([1])
        print d.data
        print e._data
        print d.data.shape
        print e._data.shape

        #self.assertEqual(d.data[0], e.data[0])
        #self.assertEqual(d.data[1], e.data[1])
        self.assertEqual(d.variables, e.variables)
        self.assertEqual(d.axis_to_variable, e.axis_to_variable)
        assert_array_almost_equal(d.data, e.data)

    def test_marginalize_larger(self):
        a = DiscreteFactor([(0, 2), (4, 3), (20, 2)])
        a._data[0, 0, 0] = 1
        a._data[0, 0, 1] = 2
        a._data[0, 1, 0] = 5
        a._data[0, 1, 1] = 8
        a._data[0, 2, 0] = 9
        a._data[0, 2, 1] = 10

        a._data[1, 0, 0] = 11
        a._data[1, 0, 1] = 12
        a._data[1, 1, 0] = 15
        a._data[1, 1, 1] = 18
        a._data[1, 2, 0] = 19
        a._data[1, 2, 1] = 21

        c = DiscreteFactor([(0, 2)])
        c._data[0] = 35
        c._data[1] = 96

        b = a.marginalize([0])
        print b.data
        print c._data
        print b.data.shape
        print c._data.shape
        self.assertEqual(b.variables, c.variables)
        self.assertEqual(b.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(b.data, c.data)

        e = DiscreteFactor([(4, 3), (20, 2)])
        e._data[0, 0] = 12
        e._data[0, 1] = 14
        e._data[1, 0] = 20
        e._data[1, 1] = 26
        e._data[2, 0] = 28
        e._data[2, 1] = 31

        d = a.marginalize([4, 20])
        print d.data
        print e._data
        print d.data.shape
        print e._data.shape
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

    def test_set_evidence_not_normalized_inplace(self):
        a = DiscreteFactor([(1, 2), (4, 3)], data=np.array(range(6)).reshape(2, 3))
        print a._data
        a.set_evidence({1: 1})
        c_data = np.array([[0, 0, 0], [3, 4, 5]])
        c = DiscreteFactor([(1, 2), (4, 3)], data=c_data)
        self.assertItemsEqual(c.variables, a.variables)
        assert_array_almost_equal(c.data, a.data)

if __name__ == '__main__':
    unittest.main()

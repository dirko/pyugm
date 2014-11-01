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

        a.multiply(b)

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

        a.multiply(e)

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

        a.multiply(b)

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

        a.multiply(b)

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

        a.multiply(b, divide=True)

        print a._data
        print a._log_normalizer
        print c._data
        print c._log_normalizer
        print a._data.shape
        print c._data.shape
        self.assertEqual(a.variables, c.variables)
        self.assertEqual(a.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(a.data, c.data)

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

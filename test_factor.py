import unittest
from factor import DiscreteFactor
from numpy.testing import assert_array_almost_equal
import numpy as np


class TestFactor(unittest.TestCase):
    def test_marginalize_small_edge(self):
        a = DiscreteFactor([(0, 2), (1, 2)])
        # a.data[0, 0] should equal 1 by default
        a.data[0, 1] = 2
        a.data[1, 0] = 5
        a.data[1, 1] = 8

        b = a.marginalize([0, 1])
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
        a.data[0, 0] = 1
        a.data[0, 1] = 2
        a.data[1, 0] = 5
        a.data[1, 1] = 8

        c = DiscreteFactor([(0, 2)])
        c.data[0] = 3
        c.data[1] = 13

        b = a.marginalize([0])
        print b.data
        print c.data
        print b.data.shape
        print c.data.shape
        self.assertEqual(b.data[0], c.data[0])
        self.assertEqual(b.data[1], c.data[1])
        self.assertEqual(b.variables, c.variables)
        self.assertEqual(b.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(b.data, c.data)

        e = DiscreteFactor([(1, 2)])
        e.data[0] = 6
        e.data[1] = 10

        d = a.marginalize([1])
        print d.data
        print e.data
        print d.data.shape
        print e.data.shape

        self.assertEqual(d.data[0], e.data[0])
        self.assertEqual(d.data[1], e.data[1])
        self.assertEqual(d.variables, e.variables)
        self.assertEqual(d.axis_to_variable, e.axis_to_variable)
        assert_array_almost_equal(d.data, e.data)

    def test_marginalize_larger(self):
        a = DiscreteFactor([(0, 2), (4, 3), (20, 2)])
        a.data[0, 0, 0] = 1
        a.data[0, 0, 1] = 2
        a.data[0, 1, 0] = 5
        a.data[0, 1, 1] = 8
        a.data[0, 2, 0] = 9
        a.data[0, 2, 1] = 10

        a.data[1, 0, 0] = 11
        a.data[1, 0, 1] = 12
        a.data[1, 1, 0] = 15
        a.data[1, 1, 1] = 18
        a.data[1, 2, 0] = 19
        a.data[1, 2, 1] = 21

        c = DiscreteFactor([(0, 2)])
        c.data[0] = 35
        c.data[1] = 96

        b = a.marginalize([0])
        print b.data
        print c.data
        print b.data.shape
        print c.data.shape
        self.assertEqual(b.variables, c.variables)
        self.assertEqual(b.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(b.data, c.data)

        e = DiscreteFactor([(4, 3), (20, 2)])
        e.data[0, 0] = 12
        e.data[0, 1] = 14
        e.data[1, 0] = 20
        e.data[1, 1] = 26
        e.data[2, 0] = 28
        e.data[2, 1] = 31

        d = a.marginalize([4, 20])
        print d.data
        print e.data
        print d.data.shape
        print e.data.shape
        self.assertEqual(d.variables, e.variables)
        self.assertEqual(d.axis_to_variable, e.axis_to_variable)
        assert_array_almost_equal(d.data, e.data)

    def test_multiply_small_inplace(self):
        a = DiscreteFactor([(0, 2), (1, 2)])
        a.data[0, 0] = 1
        a.data[0, 1] = 2
        a.data[1, 0] = 5
        a.data[1, 1] = 6

        b = DiscreteFactor([(1, 2)])
        b.data[0] = 2
        b.data[1] = 3

        c = DiscreteFactor([(0, 2), (1, 2)])
        c.data[0, 0] = 2
        c.data[0, 1] = 6
        c.data[1, 0] = 10
        c.data[1, 1] = 18

        a.multiply(b)

        print a.data
        print c.data
        print a.data.shape
        print c.data.shape
        self.assertEqual(a.variables, c.variables)
        self.assertEqual(a.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(a.data, c.data)

    def test_multiply_small_a(self):
        a = DiscreteFactor([(0, 2), (1, 2)])
        a.data[0, 0] = 1
        a.data[0, 1] = 2
        a.data[1, 0] = 5
        a.data[1, 1] = 6

        e = DiscreteFactor([(0, 2)])
        e.data[0] = 2
        e.data[1] = 3

        f = DiscreteFactor([(0, 2), (1, 2)])
        f.data[0, 0] = 1 * 2
        f.data[0, 1] = 2 * 2
        f.data[1, 0] = 5 * 3
        f.data[1, 1] = 6 * 3

        g = a.multiply(e, update_inplace=False)

        print g.data
        print f.data
        print g.data.shape
        print f.data.shape
        self.assertEqual(g.variables, f.variables)
        self.assertEqual(g.axis_to_variable, f.axis_to_variable)
        assert_array_almost_equal(g.data, f.data)

    def test_multiply_larger(self):
        a = DiscreteFactor([(0, 2), (3, 2), (12, 3)])
        a.data[0, 0, 0] = 2
        a.data[0, 0, 1] = 1
        a.data[0, 0, 2] = 2
        a.data[0, 1, 0] = 3
        a.data[0, 1, 1] = 7
        a.data[0, 1, 2] = 4

        a.data[1, 0, 0] = 1
        a.data[1, 0, 1] = 1
        a.data[1, 0, 2] = 3
        a.data[1, 1, 0] = 4
        a.data[1, 1, 1] = 9
        a.data[1, 1, 2] = 10

        b = DiscreteFactor([(0, 2), (12, 3)])
        b.data[0, 0] = 2
        b.data[0, 1] = 3
        b.data[0, 2] = 1
        b.data[1, 0] = 5
        b.data[1, 1] = 1
        b.data[1, 2] = 7

        c = DiscreteFactor([(0, 2), (3, 2), (12, 3)])
        c.data[0, 0, 0] = 2 * 2
        c.data[0, 0, 1] = 1 * 3
        c.data[0, 0, 2] = 2 * 1
        c.data[0, 1, 0] = 3 * 2
        c.data[0, 1, 1] = 7 * 3
        c.data[0, 1, 2] = 4 * 1

        c.data[1, 0, 0] = 1 * 5
        c.data[1, 0, 1] = 1 * 1
        c.data[1, 0, 2] = 3 * 7
        c.data[1, 1, 0] = 4 * 5
        c.data[1, 1, 1] = 9 * 1
        c.data[1, 1, 2] = 10 * 7

        d = a.multiply(b, update_inplace=False)

        print d.data
        print c.data
        print d.data.shape
        print c.data.shape
        self.assertEqual(d.variables, c.variables)
        self.assertEqual(d.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(d.data, c.data)

    def test_multiply_larger_correct_order(self):
        a = DiscreteFactor([(0, 2), (3, 2), (12, 3)])
        a.data[0, 0, 0] = 2
        a.data[0, 0, 1] = 1
        a.data[0, 0, 2] = 2
        a.data[0, 1, 0] = 3
        a.data[0, 1, 1] = 7
        a.data[0, 1, 2] = 4

        a.data[1, 0, 0] = 1
        a.data[1, 0, 1] = 1
        a.data[1, 0, 2] = 3
        a.data[1, 1, 0] = 4
        a.data[1, 1, 1] = 9
        a.data[1, 1, 2] = 10

        b = DiscreteFactor([(12, 3), (0, 2)])
        b.data[0, 0] = 2
        b.data[1, 0] = 3
        b.data[2, 0] = 1
        b.data[0, 1] = 5
        b.data[1, 1] = 1
        b.data[2, 1] = 7

        c = DiscreteFactor([(0, 2), (3, 2), (12, 3)])
        c.data[0, 0, 0] = 2 * 2
        c.data[0, 0, 1] = 1 * 3
        c.data[0, 0, 2] = 2 * 1
        c.data[0, 1, 0] = 3 * 2
        c.data[0, 1, 1] = 7 * 3
        c.data[0, 1, 2] = 4 * 1

        c.data[1, 0, 0] = 1 * 5
        c.data[1, 0, 1] = 1 * 1
        c.data[1, 0, 2] = 3 * 7
        c.data[1, 1, 0] = 4 * 5
        c.data[1, 1, 1] = 9 * 1
        c.data[1, 1, 2] = 10 * 7

        d = a.multiply(b, update_inplace=False)

        print d.data
        print c.data
        print d.data.shape
        print c.data.shape
        self.assertEqual(d.variables, c.variables)
        self.assertEqual(d.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(d.data, c.data)
        pass

    def test_divide_small(self):
        a = DiscreteFactor([(0, 2), (1, 2)])
        a.data[0, 0] = 1.0
        a.data[0, 1] = 2.0
        a.data[1, 0] = 5.0
        a.data[1, 1] = 6.0

        b = DiscreteFactor([(1, 2)])
        b.data[0] = 2.0
        b.data[1] = 3.0

        c = DiscreteFactor([(0, 2), (1, 2)])
        c.data[0, 0] = 1.0 / 2.0
        c.data[0, 1] = 2.0 / 3.0
        c.data[1, 0] = 5.0 / 2.0
        c.data[1, 1] = 6.0 / 3.0

        d = a.multiply(b, divide=True, update_inplace=False)

        print d.data
        print c.data
        print d.data.shape
        print c.data.shape
        self.assertEqual(d.variables, c.variables)
        self.assertEqual(d.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(d.data, c.data)

    def test_get_potential_single(self):
        a = DiscreteFactor([(4, 2), (8, 3)], data=np.array(range(6)).reshape(2, 3))
        b = a.get_potential([(8, 0), (4, 1), (2, 4)])
        self.assertAlmostEqual(b, 3)

    def test_get_potential_slice(self):
        a = DiscreteFactor([(4, 2), (8, 3)], data=np.array(range(6)).reshape(2, 3))
        b = a.get_potential([(8, 0), (9, 1), (2, 4)])
        assert_array_almost_equal(b, np.array([0, 3]))

    def test_set_evidence_not_normalized_not_inplace(self):
        a = DiscreteFactor([(1, 2), (4, 3)], data=np.array(range(6)).reshape(2, 3))
        print a.data
        b = a.set_evidence([(1, 1)])
        c_data = np.array([[0, 0, 0], [3, 4, 5]])
        c = DiscreteFactor([(1, 2), (4, 3)], data=c_data)
        self.assertItemsEqual(c.variables, b.variables)
        assert_array_almost_equal(c.data, b.data)

    def test_set_evidence_normalized_not_inplace(self):
        a = DiscreteFactor([(1, 2), (4, 3)], data=np.array(range(6)).reshape(2, 3))
        print a.data
        b = a.set_evidence([(1, 1)], normalize=True)
        c_data = np.array([[0, 0, 0], [3, 4, 5]]) / 12.0
        print c_data
        c = DiscreteFactor([(1, 2), (4, 3)], data=c_data)
        self.assertItemsEqual(c.variables, b.variables)
        assert_array_almost_equal(c.data, b.data)

    def test_set_evidence_not_normalized_inplace(self):
        a = DiscreteFactor([(1, 2), (4, 3)], data=np.array(range(6)).reshape(2, 3))
        print a.data
        b = a.set_evidence([(1, 1)], inplace=True)
        c_data = np.array([[0, 0, 0], [3, 4, 5]])
        c = DiscreteFactor([(1, 2), (4, 3)], data=c_data)
        self.assertEqual(a, b)
        self.assertItemsEqual(c.variables, a.variables)
        assert_array_almost_equal(c.data, a.data)

if __name__ == '__main__':
    unittest.main()

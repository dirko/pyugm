import unittest
from infer import Factor
from numpy.testing import assert_array_almost_equal
import numpy as np


class TestFactor(unittest.TestCase):
    def test_marginalize_small(self):
        a = Factor([(0, 2), (1, 2)])
        a.data[0, 0] = 1
        a.data[0, 1] = 2
        a.data[1, 0] = 5
        a.data[1, 1] = 8

        c = Factor([(0, 2)])
        c.data[0] = 6
        c.data[1] = 10

        b = a.marginalize([0])
        #assert_array_almost_equal(b.data, c.data, 100)
        #np.
        print b.data[0]
        print c.data[0]
        self.assertEqual(b.data[0], c.data[0])
        self.assertEqual(b.data[1], c.data[1])


if __name__ == '__main__':
    unittest.main()

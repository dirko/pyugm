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
from pyugm.infer_message import Inference
from pyugm.factor import DiscreteBelief
from pyugm.tests.test_utils import GraphTestCase


class TestInfer(GraphTestCase):
    @staticmethod
    def test_set_evidence():
        a = DiscreteFactor([1, 2, 3], np.array(range(0, 8)).reshape((2, 2, 2)))
        b = DiscreteFactor([2, 3, 4], np.array(range(1, 9)).reshape((2, 2, 2)))
        model = Model([a, b])
        inference = Inference(model)
        evidence = {2: 1, 4: 0}
        inference.set_evidence(evidence)

        c = DiscreteBelief(variables=[1, 2, 3], data=np.array(range(0, 8)).reshape((2, 2, 2)))
        c.set_evidence(evidence)
        d = DiscreteBelief(variables=[2, 3, 4], data=np.array(range(1, 9)).reshape((2, 2, 2)))
        d.set_evidence(evidence)

        assert_array_almost_equal(c.data, inference.beliefs[model.factors[0]].data)
        assert_array_almost_equal(d.data, inference.beliefs[model.factors[1]].data)

    @staticmethod
    def test_set_parameters():
        a = DiscreteFactor([1, 2], parameters=np.array([[1, 2], ['a', 0.0]], dtype=object))
        b = DiscreteFactor([2, 3], parameters=np.array([['b', 'c'], ['d', 'a']]))
        model = Model([a, b])
        print a.parameters
        new_parameters = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        inference = Inference(model)
        inference.set_parameters(new_parameters)

        c = DiscreteFactor([1, 2], np.array([1, 2, np.exp(1), 0]).reshape((2, 2)))
        d = DiscreteFactor([2, 3], np.array([np.exp(2), np.exp(3), np.exp(4), np.exp(1)]).reshape((2, 2)))

        assert_array_almost_equal(c.data, inference.beliefs[model.factors[0]].data)
        assert_array_almost_equal(d.data, inference.beliefs[model.factors[1]].data)


if __name__ == '__main__':
    unittest.main()

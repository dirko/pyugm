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
from pyugm.infer_sampling import GibbsSamplingInference
from pyugm.infer_sampling import sample_categorical
from pyugm.infer_message import ExhaustiveEnumeration
from pyugm.model import Model
from pyugm.tests.test_utils import GraphTestCase


class TestGibbsSampling(unittest.TestCase):
    def test_sample_single_variable(self):
        potential_data = np.array([0.4, 0.6])
        a = DiscreteFactor([(1, 2)], data=potential_data)

        model = Model([a])
        sampler = GibbsSamplingInference(model, samples=100000)
        sampler.calibrate()

        for belief in sampler.get_marginals(1):
            print belief.data
            assert_array_almost_equal(belief.data, potential_data, decimal=2)

    def test_sample_loopy(self):
        a = DiscreteFactor([0, 1], data=np.array([[1, 2], [2, 2]], dtype=np.float64))
        b = DiscreteFactor([1, 2], data=np.array([[3, 2], [1, 2]], dtype=np.float64))
        c = DiscreteFactor([2, 0], data=np.array([[1, 2], [3, 4]], dtype=np.float64))

        model = Model([a, b, c])
        inference = GibbsSamplingInference(model, samples=10000)
        inference.calibrate()

        exact_inference = ExhaustiveEnumeration(model)
        exhaustive_answer = exact_inference.calibrate().belief

        for factor in model.factors:
            print factor, np.sum(factor.data)
        for var in model.variables_to_factors.keys():
            print var, exhaustive_answer.marginalize([var]).data
        print
        for var in model.variables_to_factors.keys():
            print var, inference.get_marginals(var)[0].data

        print
        print 'asserting'
        for variable in model.variables:
            for belief in inference.get_marginals(variable):
                expected_table = exhaustive_answer.marginalize([variable])
                actual_table = belief.marginalize([variable])
                print expected_table.data
                print actual_table.data
                print belief.data
                assert_array_almost_equal(expected_table.normalized_data, actual_table.normalized_data, decimal=2)

    def test_sample_loopy_evidence(self):
        a = DiscreteFactor([0, 1], data=np.array([[1, 2], [2, 2]], dtype=np.float64))
        b = DiscreteFactor([1, 2], data=np.array([[3, 2], [1, 2]], dtype=np.float64))
        c = DiscreteFactor([2, 0], data=np.array([[1, 2], [3, 4]], dtype=np.float64))

        evidence = {0: 0}

        model = Model([a, b, c])
        inference = GibbsSamplingInference(model, samples=10000)
        inference.calibrate(evidence)

        exact_inference = ExhaustiveEnumeration(model)
        exhaustive_answer = exact_inference.calibrate(evidence).belief

        for variable in model.variables:
            for factor in inference.get_marginals(variable):
                expected_table = exhaustive_answer.marginalize([variable])
                actual_table = factor.marginalize([variable])
                assert_array_almost_equal(expected_table.normalized_data, actual_table.normalized_data, decimal=2)

        expected_ln_Z = np.log(exhaustive_answer.data.sum())
        self.assertAlmostEqual(expected_ln_Z, inference.partition_approximation(), places=3)

    def test_sample_categorical(self):
        counts = np.array([0, 0, 0.0])
        sample_distribution = np.array([10, 59, 1.0])
        n_points = 100000
        for i in xrange(n_points):
            sample = sample_categorical(sample_distribution)
            counts[sample] += 1
        assert_array_almost_equal(counts / n_points, sample_distribution / sum(sample_distribution), decimal=2)

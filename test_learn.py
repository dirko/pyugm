import unittest
from factor import DiscreteFactor
from infer import Model, LoopyBeliefUpdateInference
import numpy as np
from learn import LearnMRFParameters
from numpy.testing import assert_array_almost_equal


class TestLearnMRFParameters(unittest.TestCase):
    def test_get_log_likelihood(self):
        a = DiscreteFactor([1, 2], parameters=np.array([[1, 2.0], [3, 4]]))
        b = DiscreteFactor([2, 3], parameters=np.array([[3, 4.0], [5, 7]]))
        # 1 2 3  |
        # -------+----------
        # 0 0 0  | 1 * 3 = 3
        # 0 0 1  | 1 * 4 = 4
        # 0 1 0  | 2 * 5 = 10
        # 0 1 1  | 2 * 7 = 14
        # 1 0 0  | 3 * 3 = 9
        # 1 0 1  | 3 * 4 = 12
        # 1 1 0  | 4 * 5 = 20
        # 1 1 1  | 4 * 7 = 28
        #
        # p(1=0, 2, 3=1) = [4, 14] / 100
        # => p(1=0, 3=1) = 18 / 100

        model = Model([a, b])
        evidence = {1: 0, 3: 1}
        model.build_graph()

        inference = LoopyBeliefUpdateInference(model)
        c = inference.exhaustive_enumeration()
        print c
        print c.data
        print c.get_potential(evidence.items())

        learner = LearnMRFParameters(model)

        actual_log_likelihood = learner.evaluate_log_likelihood(evidence)
        print actual_log_likelihood, np.log(0.18)
        self.assertAlmostEqual(actual_log_likelihood, np.log(0.18))

    def test_get_log_likelihood_by_parameters(self):
        a = DiscreteFactor(['1', '2'], parameters=np.array([['a', 'b'], ['c', 'd']]))
        b = DiscreteFactor(['2', '3'], parameters=np.array([['e', 'f'], ['g', 'h']]))

        model = Model([a, b])
        D = len(model.parameters_to_index)
        parameters = np.zeros(D)
        parameter_out_of_order = [1, 2, 3, 4, 3, 4, 5, 7]
        parameter_names = [c for c in 'abcdefgh']
        for param, param_name in zip(parameter_out_of_order, parameter_names):
            parameters[model.parameters_to_index[param_name]] = np.log(param)
        evidence = {'1': 0, '3': 1}
        prior_sigma2 = 2.3
        model.build_graph()

        learner = LearnMRFParameters(model, prior=1.0/(prior_sigma2 ** 1.0))
        learner.parameters = parameters

        actual_log_likelihood = learner.evaluate_log_likelihood(evidence)

        prior_factor = D * (-0.5 * np.log((2.0 * np.pi * prior_sigma2)))
        print 'pn', prior_factor, D * -0.5 * np.log(prior_sigma2)
        prior_factor += sum([-0.5 / (prior_sigma2) * param ** 2.0 for param in parameters])
        print 'dim', D
        print actual_log_likelihood, np.log(0.18) + prior_factor, prior_factor
        self.assertAlmostEqual(actual_log_likelihood, np.log(0.18) + prior_factor)

    def test_learn_binary(self):
        tc1 = 31
        tc2 = 3
        obs = [DiscreteFactor([(i, 2)], parameters=np.array(['m0', 'm1'])) for i in xrange(tc1 + tc2)]
        model = Model(obs)
        evidence = dict((i, 0 if i < tc1 else 1) for i in xrange(tc1 + tc2))
        print 'evidence', evidence
        model.build_graph()
        print sorted(model.edges, key=lambda x: str(x))

        learner = LearnMRFParameters(model, prior=1.0)

        def nlog_posterior(x):
            c1 = tc1
            c2 = tc2
            x = x.reshape((2, 1))
            N = np.zeros((2, 1))
            Lamb = np.array([[1.0, 0], [0, 1.0]])
            lp = -0.5 * np.dot(np.dot((x - N).T, Lamb), (x - N))
            lp += -0.5 * x.shape[0] * np.log(2.0 * np.pi)
            lp += +0.5 * np.log(np.linalg.det((Lamb)))
            lp += np.dot(x.T, np.array([c1, c2])) - (c1 + c2) * np.log(sum(np.exp(x)))
            return -lp[0, 0]

        import scipy.optimize
        x0 = np.zeros(2)
        print 'zeros', x0
        expected_ans = scipy.optimize.fmin_l_bfgs_b(nlog_posterior, x0, approx_grad=True)
        #x0 = np.ones(2) * 0.000
        actual_ans = learner.fit_without_gradient(evidence).ans#, x0)
        print actual_ans
        print expected_ans
        assert_array_almost_equal(actual_ans[0], expected_ans[0])
        self.assertAlmostEqual(actual_ans[1], expected_ans[1])


import unittest
from factor import DiscreteFactor
from infer import Model, LoopyBeliefUpdateInference, DistributeCollectProtocol, FloodingProtocol
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

    def test_evaluate_derivative(self):
        D = 8
        delta = 10.0**-10
        for variable_index in range(D):
            a = DiscreteFactor(['1', '2'], parameters=np.array([['a', 'b'], ['c', 'd']]))
            b = DiscreteFactor(['2', '3'], parameters=np.array([['e', 'f'], ['g', 'h']]))
            c = DiscreteFactor(['3', '4'], parameters=np.array([['i', 'j'], ['k', 'l']]))

            model = Model([a, b, c])
            D = len(model.parameters_to_index)

            delta_vector = np.zeros(D)
            delta_vector[variable_index] = delta

            parameters = np.zeros(D)
            parameters_plus_delta = np.zeros(D)
            parameter_out_of_order = [1, 2, 3, 4, 3, 4, 5, 7, 8, 9, 10, 11]
            parameter_names = [c for c in 'abcdefghijkl']
            for param, param_name in zip(parameter_out_of_order, parameter_names):
                parameters[model.parameters_to_index[param_name]] = np.log(param)
                parameters_plus_delta[model.parameters_to_index[param_name]] = np.log(param)
            parameters_plus_delta += delta_vector

            evidence = {'1': 0, '3': 1}
            prior_sigma2 = 2.3
            model.build_graph()

            learner = LearnMRFParameters(model, prior=1.0/(prior_sigma2 ** 1.0))
            learner.parameters = parameters
            actual_log_likelihood1, actual_derivative1 = learner.evaluate_log_likelihood_and_derivative(evidence)

            learner.parameters = parameters_plus_delta
            actual_log_likelihood2, actual_derivative2 = learner.evaluate_log_likelihood_and_derivative(evidence)

            expected_deriv = (actual_log_likelihood2 - actual_log_likelihood1) / delta  # * delta_vector / delta / delta

            prior_factor = D * (-0.5 * np.log((2.0 * np.pi * prior_sigma2)))
            print 'pn', prior_factor, D * -0.5 * np.log(prior_sigma2)
            prior_factor += sum([-0.5 / (prior_sigma2) * param ** 2.0 for param in parameters])
            print 'dim', D
            print actual_log_likelihood1, np.log(0.18) + prior_factor, prior_factor
            #self.assertAlmostEqual(actual_log_likelihood1, np.log(0.18) + prior_factor)
            #self.assertAlmostEqual(actual_log_likelihood2, np.log(0.18) + prior_factor, delta=10.0**-3)

            print 'derivs'
            print actual_derivative1
            print actual_derivative2
            print expected_deriv
            self.assertAlmostEqual(expected_deriv, actual_derivative1[variable_index], delta=10.0**-4)

    def test_learn_without_gradient_binary(self):
        tc1 = 70
        tc2 = 14
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
        expected_ans = scipy.optimize.fmin_l_bfgs_b(nlog_posterior, x0, approx_grad=True, pgtol=10.0**-10)
        #x0 = np.ones(2) * 0.000
        actual_ans = learner.fit_without_gradient(evidence).ans#, x0)
        print actual_ans
        print expected_ans
        self.assertAlmostEqual(actual_ans[1], expected_ans[1])
        assert_array_almost_equal(actual_ans[0], expected_ans[0], decimal=4)

    def test_learn_with_gradient_binary(self):
        tc1 = 70
        tc2 = 14
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
        expected_ans = scipy.optimize.fmin_l_bfgs_b(nlog_posterior, x0, approx_grad=True, pgtol=10.0**-10)
        #x0 = np.ones(2) * 0.000
        actual_ans = learner.fit(evidence).ans#, x0)
        print actual_ans
        print expected_ans
        self.assertAlmostEqual(actual_ans[1], expected_ans[1])
        assert_array_almost_equal(actual_ans[0], expected_ans[0], decimal=5)

    def test_compare_gradientless_and_gradient_learning(self):
        seq = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
        seqh = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
        factors = []
        hidden_factors = []
        evidence = {}
        o_parameters=np.array([['a', 'b'], ['c', 'd']])
        h_parameters=np.array([['e', 'f'], ['g', 'h']])
        for i, (s, sh) in enumerate(zip(seq, seqh)):
            obs = DiscreteFactor(['o_{}'.format(i), 'h_{}'.format(i)], parameters=o_parameters)
            evidence['o_{}'.format(i)] = s
            evidence['h_{}'.format(i)] = sh
            factors.append(obs)
            if i < len(seq):
                hid = DiscreteFactor(['h_{}'.format(i), 'h_{}'.format(i + 1)], parameters=h_parameters)
                factors.append(hid)
                hidden_factors.append(hid)

        model = Model(factors)
        model.build_graph()
        update_order = DistributeCollectProtocol(model)
        learn = LearnMRFParameters(model, update_order=update_order)
        learn.fit(evidence)
        print learn.ans
        #print learn.iterations
        ans1 = learn.ans[:2]

        print
        update_order = DistributeCollectProtocol(model)
        learn = LearnMRFParameters(model, update_order=update_order)
        learn.fit_without_gradient(evidence)
        print learn.ans
        #print learn.iterations
        ans2 = learn.ans[:2]
        assert_array_almost_equal(ans1[1], ans2[1])
        assert_array_almost_equal(ans1[0], ans2[0])

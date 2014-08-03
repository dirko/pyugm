from factor import DiscreteFactor
from infer import Model, LoopyBeliefUpdateInference
import numpy as np


def example_fully_specified_run():
    seq = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    emissions = np.zeros((2, 2))
    emissions[0, 0] = 0.8
    emissions[0, 1] = 0.2
    emissions[1, 0] = 0.3
    emissions[1, 1] = 0.7

    transitions = np.zeros((2, 2))
    transitions[0, 0] = 0.9
    transitions[0, 1] = 0.1
    transitions[1, 0] = 0.2
    transitions[1, 1] = 0.8

    a = DiscreteFactor([0], np.array([0.5, 0.5]))
    factors = [a]
    for i, s in enumerate(seq):
        a = DiscreteFactor([i, i + 100], np.array(emissions))
        a.set_evidence([(i + 100, s)], normalize=True, inplace=True)
        factors.append(a)

        if i != 0:
            b = DiscreteFactor([i - 1, i], np.array(transitions))
            factors.append(b)

    model = Model(factors)
    model.build_graph()

    for factor in model.factors:
        print factor, factor.data

    inference = LoopyBeliefUpdateInference(model)

    inference.set_up_belief_update()
    change = inference.update_beliefs(number_of_updates=185)
    print change
    for factor in model.factors:
        print factor, factor.data

    for i in xrange(0, len(seq)):
        for factor in model.get_marginals(i, normalize=False):
            print i, factor.data, factor.data.sum()
    for i, s in enumerate(seq):
        print i, s, model.get_marginals(i)[0].data


def example_learning_run():
    seqs = [[1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]]

    # (Disregard:)
    # A _|_ B      <=> p(A, B) = p(A)p(B)
    # A _|_ B | C  <=> p(A, B | C) = p(A|C)p(B|C) = p(A,C)p(B,C)/(p(C)^2)
    # p(A, B | C, D) = p(C,D|A,B)*p(A,B)/p(C,D) = p(C|A,B)p(D|A,B)*p(A,B)/p(C,D) = p(C,A,B)p(D,A,B)/p(C,D)
    #                = p(C,A,B)p(D,A,B)/(p(C)p(D)) = p(A,B|C)p(A,B|D)  <= C _|_ D ,  A,B _|_ C|D ,  A,B _|_ D|C

    parameter_discrete = 10
    emissions = np.zeros((2, 2, parameter_discrete, parameter_discrete))
    for itheta1 in xrange(parameter_discrete):
        for itheta2 in xrange(parameter_discrete):
            theta1 = 1.0 * (itheta1 + 1) / (parameter_discrete + 1)
            theta2 = 1.0 * (itheta2 + 1) / (parameter_discrete + 1)
            emissions[0, 0, itheta1, itheta2] = theta1
            emissions[0, 1, itheta1, itheta2] = 1 - theta1
            emissions[1, 0, itheta1, itheta2] = theta2
            emissions[1, 1, itheta1, itheta2] = 1 - theta2

    transitions = np.zeros((2, 2, parameter_discrete, parameter_discrete))
    for itheta1 in xrange(parameter_discrete):
        for itheta2 in xrange(parameter_discrete):
            theta1 = 1.0 * (itheta1 + 1) / (parameter_discrete + 1)
            theta2 = 1.0 * (itheta2 + 1) / (parameter_discrete + 1)
            transitions[0, 0, itheta1, itheta2] = theta1
            transitions[0, 1, itheta1, itheta2] = 1 - theta1
            transitions[1, 0, itheta1, itheta2] = theta2
            transitions[1, 1, itheta1, itheta2] = 1 - theta2

    theta1 = DiscreteFactor([(1000, parameter_discrete)], normalize=True)  # transitions
    theta2 = DiscreteFactor([(1001, parameter_discrete)], normalize=True)
    lambd1 = DiscreteFactor([(2000, parameter_discrete)], normalize=True)  # emissions
    lambd2 = DiscreteFactor([(2001, parameter_discrete)], normalize=True)

    a = DiscreteFactor([0], np.array([0.5, 0.5]))
    factors = [theta1, theta2, lambd1, lambd2, a]
    for i, s in enumerate(seqs[0]):
        a = DiscreteFactor([(i, 2), (i + 100, 2), (2000, parameter_discrete), (2001, parameter_discrete)], np.array(emissions))
        a.set_evidence([(i + 100, s)], normalize=True, inplace=True)
        factors.append(a)

        if i != 0:
            b = DiscreteFactor([(i - 1, 2), (i, 2),
                        (1000, parameter_discrete), (1001, parameter_discrete)], np.array(transitions))
            factors.append(b)

    model = Model(factors)
    model.build_graph()

    print 'Factors'
    for factor in model.factors:
        print factor  # , factor.data

    print '\nEdges'
    for edge in model.edges:
        print edge[0], '->', edge[1]

    inference = LoopyBeliefUpdateInference(model)

    inference.set_up_belief_update()
    change = inference.update_beliefs(number_of_updates=15)
    print change
    for factor in model.factors:
        print factor, factor.data

    for i in xrange(0, len(seqs[0])):
        for factor in model.get_marginals(i, normalize=False):
            print i, factor.data, factor.data.sum()
    for i, s in enumerate(seqs[0]):
        print i, s, model.get_marginals(i)[0].data

if __name__ == '__main__':
    example_learning_run()

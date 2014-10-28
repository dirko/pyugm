import numpy as np

from pyugm.factor import DiscreteFactor
from infer import LoopyBeliefUpdateInference
from model import Model
from learn import LearnMRFParameters


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

    for factor in model.factors:
        print factor, factor.data

    inference = LoopyBeliefUpdateInference(model)

    change = inference.calibrate(number_of_updates=185)
    print change
    for factor in model.factors:
        print factor, factor.data

    for i in xrange(0, len(seq)):
        for factor in model.get_marginals(i, normalize=False):
            print i, factor.data, factor.data.sum()
    for i, s in enumerate(seq):
        print i, s, model.get_marginals(i)[0].data


def example_learning_run():
    seqs = [[1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1]]
    factors = []
    evidence = {}
    for s, seq in enumerate(seqs):
        for e, elem in enumerate(seq):
            factors.append(DiscreteFactor([e + 100 * s, e + 1 + 100 * s],
                                          parameters=np.array([['a', 'b'], ['c', 'd']])))
            factors.append(DiscreteFactor([e + 100 * s, e + 100 * s + 10000],
                                          parameters=np.array([['e', 'f'], ['g', 'h']])))
            evidence[e + 100 * s + 10000] = elem
    model = Model(factors)
    learner = LearnMRFParameters(model, prior=1.0)
    learner.fit(evidence)
    print learner.ans

if __name__ == '__main__':
    example_learning_run()

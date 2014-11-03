#!/bin/python
import numpy as np

from pyugm.factor import DiscreteFactor
from pyugm.model import Model
from pyugm.learn import LearnMRFParameters


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
                                          parameters=np.array([['trans00', 'trans01'], ['trans10', 'trans11']])))
            factors.append(DiscreteFactor([e + 100 * s, e + 100 * s + 10000],
                                          parameters=np.array([['obs00', 'obs01'], ['obs10', 'obs11']])))
            evidence[e + 100 * s + 10000] = elem
    model = Model(factors)
    learner = LearnMRFParameters(model, prior=1.0)
    learner.fit(evidence)
    print learner.parameters

if __name__ == '__main__':
    example_learning_run()

from factor import DiscreteFactor
from infer import Model, LoopyBeliefUpdateInference
import numpy as np


class LearnMRFParameters():
    """
    Find an approximation to the posterior given a model and prior.
    """

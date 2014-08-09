from factor import DiscreteFactor
from infer import Model, LoopyBeliefUpdateInference
import numpy as np


class LearnMRFParameters:
    """
    Find an approximation to the posterior given a model and prior.
    """
    def __init__(self, model, evidence, prior=1.0):
        self.model = model
        self.evidence = evidence
        self.parameters = np.zeros(len(self.model.parameters_to_index))

    def evaluate_objective_derivative(self):
        #self.model.set_parameters(self.parameters)
        #self.model.set_evidence(evidence=self.evidence)
        #inference = LoopyBeliefUpdateInference(self.model)
        #inference.set_up_belief_update()
        #change = inference.update_beliefs(number_of_updates=35)
        #Z_observed = self.model.factors[0].data.sum()
        #empirical_expected_features = self.model.get_expected_parameters()

        self.model.set_parameters(self.parameters)
        inference = LoopyBeliefUpdateInference(self.model)
        inference.set_up_belief_update()
        change = inference.update_beliefs(number_of_updates=35)
        Z_total = self.model.factors[0].data.sum()
        #model_expected_features = self.model.get_expected_parameters()
        log_likelihood = np.log(Z_observed) - np.log(Z_total)

        #gradient = empirical_expected_features - model_expected_features

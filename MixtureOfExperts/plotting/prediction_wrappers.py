"""
This file contains wrappers for the plotting functions of the models included within this package.

These functions handle probit models. Adding these wrappers, rather than adding to model._predict... functions, allows
us to also do predictions in latent feature space should we wish. It also saves repeating code across every model.
"""

from __future__ import division

def predict_expectation(self):
    """
    A wrapper around model._predict_expectation which deals with probit models
    """
    raise NotImplementedError

def predict_marginal_expectation(self):
    """
    A wrapper around model._predict_marginal_expectation which deals with probit models
    """
    raise NotImplementedError

def predict_density(self):
    """
    A wrapper around model._predict_density which deals with probit models
    """
    raise NotImplementedError

def predict_density_marginal(self):
    """
    A wrapper around model._predict_density_marginal which deals with probit models
    """
    raise NotImplementedError


from __future__ import division
import logging

class InputModelBase(object):
    """
    Base object for local input models.

    New local input models (with each parametric model in the exponential family) should inherit from this base class
    and over-ride the below methods. This is so missing methods can be properly handled.
    """

    def __init__(self, name):
        self.__name__ = name

    # Methods to over-write
    def log_marginal(self, x, p=None):
        """ The marginal, or joint marginal over inputs, depending on the shape of x

        :param x:       covariates
            :type:         (num_samples * covariate dimension)   OR    (num_samples * P)
        :param p:       covariates indices of x to calculate
            :type:          list
        :return:

        Two cases: pass x_p no p;                                   calculate prod across all,
                   pass all x and a list of indices;                calculate across these
        """
        raise NotImplementedError

    def sample_marginal(self, lenp, samples):
        raise NotImplementedError

    def log_predictive_marginal(self, x, X, p=None):
        raise NotImplementedError

    def sample_predictive_marginal(self, X, samples):
        raise NotImplementedError

    def plot_xlikelihood(self, x, covariate=0, path=None):
        """
        Plot the input likelihood and marginalised (conditioned on some example subsets) likelihood all in one figure.
        The subsets are shown as scatter points in the same colour. When input is multi dimensional, other covariates
        are fixed at their mean values. This leads to plotting a slice, which is important when interpreting results.

        :param covariate:       The index of the covariate we wish to plot
            :type:              int
        """
        raise NotImplementedError

    def plot_priors(self):
        raise NotImplementedError

    # Deprecated methods
    def h(self, x, p=None):
        logging.critical('Deprecated, use log_marginal()')
        raise NotImplementedError

    def h_sample(self, lenp, samples):
        logging.critical('Deprecated, use sample_marginal()')
        raise NotImplementedError

    def logh(self, x, p=None):
        """
        Evaluate the log probability density log(h(x))
        :param x:       input
        :param p:       covariates indices of x to calculate
        :return:

        Two cases: (p=None)   pass x_p and calculate prod across all
                   (p=list)   pass all x and a list of the relevant indexes to calculate across
        """
        logging.critical('Deprecated, use log_marginal()')
        raise NotImplementedError

    def hX(self, x, X, p=None):
        """
        Evaluate the probability density h(x|X)
        :param x:    The input we wish to obtain the probability density for
        :param X:    The conditional input
        :param p:    The indices we wish to compute the probability density for.
        :return:

        Two cases: pass x_p and X_p, and calculate prod across p,
                   pass all x and all X, and a list of the relevant indexes to calculate across
        """
        logging.critical('Deprecated, use log_predictive_marginal()')
        raise NotImplementedError

    def loghX(self, x, X, p=None):
        logging.critical('Deprecated, use log_predictive_marginal()')
        raise NotImplementedError

    def hX_sample(self, X, samples):
        logging.critical('Deprecated, use sample_predictive_marginal()')
        raise NotImplementedError

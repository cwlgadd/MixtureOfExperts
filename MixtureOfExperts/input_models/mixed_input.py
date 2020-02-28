"""
A class for mixtures of covariate likelihoods

Author:
    Charles Gadd
"""
from __future__ import division
import logging
logger = logging.getLogger(__name__)
import numpy as np

__all__ = ['MixedInput']

class MixedInput(object):
    """
    A class for mixed covariate likelihood, for covariates belonging to different domains.
    """

    # public (accessible through @property decorators below)
    _likelihoods = None                                     # the covariate likelihood models
    _indexes = None                                         # a list of index lists matching input_models and covariates
    _flat_indexes = None                                    # a flattened list of indexes

    def __init__(self, likelihoods, indexes, name='mixedLikelihood'):
        """
        Initialise the likelihood object
        """
        self.__name__ = name
        self._likelihoods = likelihoods

        self._flat_indexes = [item for sublist in indexes for item in sublist]
        assert len(np.unique(self._flat_indexes)) == len(self._flat_indexes)            # check no repeats
        self._indexes = indexes

    def __str__(self):
        """
        Return a string representation of the object.
        """

        s = "\nX LIKELIHOOD: " + self.__name__
        for likelihood in self._likelihoods:
            s += '\n___________________________'
            s += str(likelihood)
        s += '\n =========================='

        return s

    def log_marginal(self, x, p=None):
        """
        Evaluate the log marginal likelihood log(h(x))=\sum_p log(h(x_p)), where p is the indexing set for each likelihood

        :param x:       input
        :param p:       covariates indices of x to calculate
        :return:

        Two cases: (p=None)  pass all x and calculate prod across all P
                   (p=list)  pass all x and a list of the relevant indexes to calculate across

        Note: In other input_models we can pass a subset xp with p=None as we know they share a distribution (and x_p
              is considered contain all indexes relevant to that likelihood).
              This is NOT true here. This means the use is slightly different, and code should be written around
              this, more specific, case for generality.
        """
        assert x.ndim == 2, 'MixedInput log_marginal(): Require 2d ndarray x'
        assert x.shape[1] == self.recursive_len(self._indexes), 'MixedInput log_marginal(): Bad number of covariates.'

        if p is None:                                   # then we need all model covariates and calculate across all
            pass
        elif isinstance(p, (list,)):                    # then we need all model covariates and calculate across listed
            # check p contains only unique indices
            assert len(p) == len(np.unique(p))
            # check p is a subset of self._indexes
            assert all(index in self._flat_indexes for index in p)
        else:
            raise ValueError
        #assert np.isnan(x).any() == False

        logxlik = 0
        if p is None:                                   # Calculate for all covariates
            counter = 0
            for likelihood in self._likelihoods:
                logxlik += likelihood.log_marginal(x[:, self._indexes[counter]])

                #logging.debug('logh: covar {0}, xlik {1}, prodxlik {2}, logxlik {3}'.format(counter,
                #                                                    np.exp(likelihood.logh(x[:, self._indexes[counter]])),
                #                                                    np.exp(logxlik), logxlik))

                counter += 1
        elif isinstance(p, (list,)):                    # Calculate for only the covariates in keyword list
            for _p in p:
                # Find which likelihood index belongs to
                contains = [i for i in range(len(self._indexes)) if (_p in self._indexes[i])]
                assert len(contains) == 1, 'mL.h(): index {0} should only belong to one likelihood.'.format(_p)

                likelihood = self._likelihoods[contains[0]]
                logxlik += likelihood.log_marginal(x[:, [_p]])
        else:
            raise ValueError('mL.h(): p must be a list of indices.')

        return logxlik

    def sample_marginal(self, lenp, samples):
        """
        Sample the marginal likelihood h(x_p) for each p, where p is the indexing set of each likelihood

        :param _:               replaces lenp which is not needed for this wrapper as we have self._indexes
        :param samples:         number of samples to obtain
        :return:                array of samples from all covariates
        """
        assert lenp == len(self._flat_indexes)

        X = np.zeros((samples, len(self._flat_indexes)))

        counter = 0
        for likelihood in self._likelihoods:
            X[:, self._indexes[counter]] = likelihood.sample_marginal(len(self._indexes[counter]), samples)
            counter += 1

        return X

    def log_predictive_marginal(self, x, X, p=None):
        """
        Evaluate the log probability density log(h(x|X))
        :param x:       input
        :param X:       inputs conditioned upon
        :param p:       covariates indices of x to calculate
        :return:

        Two cases: (p=None)  pass all x and all X, and calculate prod across all P
                   (p=list)  pass all x and all X, and a list of the relevant indexes to calculate across

        Note: In other input_models we can pass a subset xp with p=None as we know they share a distribution (and x_p
              is considered contain all indexes relevant to that likelihood).
              This is NOT true here. This means the use is slightly different, and code should be written around
              this, more specific, case for generality.
        """
        assert x.ndim == X.ndim == 2, 'MixedInput log_predictive_marginal(): Require 2d ndarray x and X'
        assert x.shape[1] == X.shape[1] == self.recursive_len(self._indexes), 'MixedInput log_predictive_marginal(): ' \
                                                                              'Bad number of covariates.'

        if p is None:  # then we need all model covariates and calculate across all
            pass
        elif isinstance(p, (list,)):  # then we need all model covariates and calculate across listed
            # check p contains only unique indices
            assert len(p) == len(np.unique(p))
            # check p is a subset of self._indexes
            assert all(index in self._flat_indexes for index in p)
        else:
            raise ValueError
        #assert np.isnan(X).any() == False

        logxlik = 0
        counter = 0

        if p is None:  # Calculate for all covariates
            for likelihood in self._likelihoods:
                logxlik += likelihood.log_predictive_marginal(x[:, self._indexes[counter]],
                                                              X[:, self._indexes[counter]])

                #logging.debug('logh: covar {0}, xlik {1}, prodxlik {2}, logxlik {3}'.format(counter,
                #                                             np.exp(likelihood.loghX(x[:, self._indexes[counter]],
                #                                                                     X[:, self._indexes[counter]])),
                #                                             np.exp(logxlik), logxlik))

                counter += 1
        elif isinstance(p, (list,)):  # Calculate for only the covariates in keyword list
            for _p in p:
                # Get which likelihood index belongs to
                contains = [i for i in range(len(self._indexes)) if (_p in self._indexes[i])]
                assert len(contains) == 1, 'MixedInput log_predictive_marginal() ' \
                                           'index {0} appears more than once in indexes list.'.format(_p)

                likelihood = self._likelihoods[contains[0]]
                logxlik += likelihood.log_predictive_marginal(x[:, [_p]], X[:, [_p]])
        else:
            raise ValueError('MixedInput log_predictive_marginal() p must be a list of indices.')

        return logxlik

    def sample_predictive_marginal(self, X, samples):
        """
        Sample the predictive marginal likelihood h(x|X).

        :param xcond:       the matrix of covariates X we condition on
            :type           matrix Nj * xdim
        :param samples:     the number of samples we wish to take
            :type           int
        :param P:           the indices of the covariates we wish to sample
            :type           list
        :return:            the marginal likelihood density for the input - h(xi)
            :type           vector Nj
        """
        assert X.ndim == 2, 'MixedInput sample_predictive_marginal() Require 2d ndarray X'
        assert (X.shape[1] == self.recursive_len(self._indexes))  # check we have all covariates

        xsample = np.empty((samples, X.shape[1]))
        xsample[:] = np.nan

        # Calculate for all covariates
        counter = 0
        for likelihood in self._likelihoods:
            xsample[:, self._indexes[counter]] = likelihood.sample_predictive_marginal(X[:, self._indexes[counter]],
                                                                                       samples=samples)
            counter += 1

        return xsample

    def plot_xlikelihood(self, x):
        """
        Plot the input likelihood and marginalised (conditioned on some example subsets) likelihood all in one figure.
        The subsets are shown as scatter points in the same colour. When input is multi dimensional, other covariates
        are fixed at their mean values. This leads to plotting a slice, which is important when interpreting results.
        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError('cannot import matplotlib')

        assert x.shape[1] == self.recursive_len(self._indexes)

        counter = 0
        for likelihood in self._likelihoods:
            if counter > 300 and counter < 500:
                likelihood.plot_xlikelihood(x[:, self._indexes[counter]], covariate=0, path=None)
            counter += 1

    def recursive_len(self, item):
        if type(item) == list:
            return sum(self.recursive_len(subitem) for subitem in item)
        else:
            return 1



    # def h(self, x, p=None):
    #     """
    #     Evaluate the marginal likelihood h(x)=\prod_p h(x_p), where p is the indexing set for each likelihood
    #
    #     :param x:       input
    #     :param p:       covariates indices of x to calculate
    #     :return:
    #
    #     Two cases: (p=None)  pass all x and calculate prod across all P
    #                (p=list)  pass all x and a list of the relevant indexes to calculate across
    #
    #     Note: In other input_models we can pass a subset xp with p=None as we know they share a distribution (and x_p
    #           is considered contain all indexes relevant to that likelihood).
    #           This is NOT true here. This means the use is slightly different, and code should be written around
    #           this, more specific, case for generality.
    #     """
    #     assert x.ndim == 2, 'mL.h(): Require 2d ndarray x'
    #     assert x.shape[1] == self.recursive_len(self._indexes), 'mL.h(): Bad number of covariates.'
    #
    #     if p is None:                                   # then we need all model covariates and calculate across all
    #         pass
    #     elif isinstance(p, (list,)):                    # then we need all model covariates and calculate across listed
    #         # check p contains only unique indices
    #         assert len(p) == len(np.unique(p))
    #         # check p is a subset of self._indexes
    #         assert all(index in self._flat_indexes for index in p)
    #     else:
    #         raise ValueError
    #     #assert np.isnan(x).any() == False
    #
    #     xlik = 1
    #     if p is None:                                   # Calculate for all covariates
    #         counter = 0
    #         for likelihood in self._likelihoods:
    #             xlik *= likelihood.h(x[:, self._indexes[counter]])
    #
    #             # debugging - checking if log space can solve rounding errors - no.
    #             #logger.debug('h: covar {0}, xlik {1}, prodxlik {2}'.format(counter,
    #             #                                                           likelihood.h(x[:, self._indexes[counter]]),
    #             #                                                           xlik))
    #             # end debugging
    #
    #             counter += 1
    #     elif isinstance(p, (list,)):                    # Calculate for only the covariates in keyword list
    #         for _p in p:
    #             # Find which likelihood index belongs to
    #             contains = [i for i in range(len(self._indexes)) if (_p in self._indexes[i])]
    #             assert len(contains) == 1, 'mL.h(): index {0} should only belong to one likelihood.'.format(_p)
    #
    #             likelihood = self._likelihoods[contains[0]]
    #             xlik *= likelihood.h(x[:, [_p]])
    #     else:
    #         raise ValueError('mL.h(): p must be a list of indices.')
    #
    #     return xlik

    # def hX(self, x, X, p=None):
    #     """
    #     Evaluate the probability density h(x|X)
    #     :param x:       input
    #     :param X:       inputs conditioned upon
    #     :param p:       covariates indices of x to calculate
    #     :return:
    #
    #     Two cases: (p=None)  pass all x and all X, and calculate prod across all P
    #                (p=list)  pass all x and all X, and a list of the relevant indexes to calculate across
    #
    #     Note: In other input_models we can pass a subset xp with p=None as we know they share a distribution (and x_p
    #           is considered contain all indexes relevant to that likelihood).
    #           This is NOT true here. This means the use is slightly different, and code should be written around
    #           this, more specific, case for generality.
    #     """
    #     assert x.ndim == X.ndim == 2, 'mL.h(): Require 2d ndarray x'
    #     assert x.shape[1] == X.shape[1] == self.recursive_len(self._indexes), 'mL.h(): Bad number of covariates.'
    #
    #     if p is None:  # then we need all model covariates and calculate across all
    #         pass
    #     elif isinstance(p, (list,)):  # then we need all model covariates and calculate across listed
    #         # check p contains only unique indices
    #         assert len(p) == len(np.unique(p))
    #         # check p is a subset of self._indexes
    #         assert all(index in self._flat_indexes for index in p)
    #     else:
    #         raise ValueError
    #     #assert np.isnan(X).any() == False
    #
    #     xlik = 1
    #     counter = 0
    #     if p is None:  # Calculate for all covariates
    #         for likelihood in self._likelihoods:
    #             xlik *= likelihood.hX(x[:, self._indexes[counter]], X[:, self._indexes[counter]])
    #
    #             # debugging - checking if log space can solve rounding errors - no.
    #             #print('hx: covar {0}, xlik {1}, prodxlik {2}'.format(counter,
    #             #                                                 likelihood.hX(x[:, self._indexes[counter]],
    #             #                                                                        X[:, self._indexes[counter]]),
    #             #                                                 xlik))
    #             #if xlik == 0:
    #             #    print(x[:, self._indexes[counter]])
    #             #    likelihood.plot_xlikelihood(X[:, self._indexes[counter]], covariate=0, path=None)
    #             # end debugging
    #
    #             counter += 1
    #     elif isinstance(p, (list,)):  # Calculate for only the covariates in keyword list
    #         for _p in p:
    #             # Get which likelihood index belongs to
    #             contains = [i for i in range(len(self._indexes)) if (_p in self._indexes[i])]
    #             assert len(contains) == 1, 'mL.marglik: index {0} appears more than once in indexes list.'.format(_p)
    #
    #             likelihood = self._likelihoods[contains[0]]
    #             xlik *= likelihood.hX(x[:, [_p]], X[:, [_p]])
    #     else:
    #         raise ValueError('mixedLikelihood._pred_marglikelihood_x: p must be a list of indices.')
    #
    #     return xlik

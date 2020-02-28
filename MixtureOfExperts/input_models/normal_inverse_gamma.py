"""
Gaussian local input model(s) for continuous covariates taking values in the real domain.

Author:
    Charles Gadd
"""
from __future__ import division
import numpy as np
import scipy
import copy
import logging
logger = logging.getLogger(__name__)

from .base_class import InputModelBase
from ..utils.mvt_t import multivariate_t_distribution_pdf as mvt_t

__all__ = ['NormalInverseGamma']


class NormalInverseGamma(InputModelBase):
    """
    A class for Gaussian local input model with conjugate Normal-inverse Gamma prior distribution.
    """

    # public (accessible through @property decorators below)
    _u0p = None
    _cp = None
    _axp = None
    _bxp = None

    # private
    _domain = 'Reals'

    @property
    def u0p(self):
        return self._u0p

    @property
    def cp(self):
        return self._cp

    @property
    def axp(self):
        return self._axp

    @property
    def bxp(self):
        return self._bxp

    def __init__(self, u0p, cp, axp, bxp, name='NormalInverseGamma'):
        """
        Initialise the NormalInverseGamma class.

        :param u0p:             Gaussian mean hyper-parameter
        :param cp:              Gaussian variance hyper-parameter
        :param axp:             Inverse-Gamma shape hyper-parameter.
        :param bxp:             Inverse-Gamma scale hyper-parameter.
        :param name:            Input model name
        """
        super(NormalInverseGamma, self).__init__(name=name)
        self._u0p = u0p
        self._cp = cp
        self._axp = axp
        self._bxp = bxp

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = "\nGaussian local input model with conjugate NIG prior: " + self.__name__ + '\n'
        s += "\t Distribution over: {0} \n".format(str(self._domain))
        s += "\t u0p: {0} \n".format(str(self.u0p))
        s += "\t cp: {0} \n".format(str(self.cp))
        s += "\t axp: {0} \n".format(str(self.axp))
        s += "\t bxp: {0}".format(str(self.bxp))
        return s

    def log_marginal(self, x, p=None):
        """ Evaluate the marginal probability density.

            if x.shape[0] == 1:             evaluate h(x_i) = \prod h(x_{i,p})
            else:                           evaluate joint marginal.
        """
        assert x.ndim == 2
        if p is None:  # Evaluate product over all P indices
            xp = np.asmatrix(x)
        elif isinstance(p, (list,)):  # Evaluate product of only desired P indices
            assert x.shape[1] >= len(p)
            xp = copy.deepcopy(np.asmatrix(x[:, p]))
        else:
            raise ValueError('NormalInverseGamma log_marginal(): p must be a list of indices.')
        assert np.isnan(xp).any() == False

        # If we have multiple xs then return joint marginal, otherwise full marginal
        if x.shape[0] == 1:
            return self._log_full_marginal(xp)
        else:
            return self._log_joint_marginal(xp)

    def _log_full_marginal(self, x):
        """
        Calculate the marginal likelihood h(x).

        :param x:          the positions vector(s) whose full marginal likelihood densities we want to calculate
            :type           matrix Nj * xdim
        :return:            the marginal likelihood density for the input - h(xi)
            :type           vector Nj
        """
        if x.ndim == 2:
            x = x.reshape((1, -1))

        log_xlik = np.zeros((x.shape[0],))
        # debug_xlik = np.ones((x.shape[0],))
        for subp in range(x.shape[1]):
            loc = self._u0p
            Sigma = (self._bxp / self._axp) * ((self._cp + 1) / self._cp)
            dof = 2 * self._axp

            for i in range(x.shape[0]):
                log_xlik[i] += scipy.stats.t.logpdf(np.asmatrix(x[i, subp]), df=dof, loc=loc, scale=np.sqrt(Sigma))
                # debug_xlik[i] *= scipy.stats.t.pdf(np.asmatrix(x[i, subp]), df=dof, loc=loc, scale=np.sqrt(Sigma))

            assert np.isfinite(log_xlik).all(), 'NormalInverseGamma _log_full_marginal(): underflow - ' \
                                                '{0},{1},{2},{3},{4}'.format(x[:, subp], log_xlik, dof, loc, Sigma)
        # assert np.isclose(np.exp(log_xlik), debug_xlik).all()
        return log_xlik

    def _log_joint_marginal(self, X):
        """
        Calculate the joint marginal likelihood h(X).

        :param x:          the positions vector(s) whose likelihood densities we want to calculate
            :type           matrix Nj * xdim
        :return:            the joint marginal likelihood density for the input - h(X)
            :type           vector Nj
        """
        loc = self._u0p * np.ones(X.shape[0])  # u0p shared across covariates

        log_lik = 0
        for subp in range(X.shape[1]):
            Sigma = (self._bxp / self._axp) * \
                    np.linalg.inv(
                        np.eye((X.shape[0])) - (1. / (self._cp + X.shape[0])) * np.ones((X.shape[0], X.shape[0])))
            # TODO: can avoid matrix inversion in here.
            dof = 2 * self._axp

            log_lik += np.log(mvt_t(X[:, subp], mu=loc, Sigma=np.sqrt(Sigma), df=dof))

        if not np.isfinite(log_lik):
            logger.warning('Underflow in NIG._log_joint_marginal()')
        assert np.isfinite(log_lik), 'GaussNIG._joint_marg_x_(): {0},{1},{2},{3},{4}'.format(X[:, subp], log_lik, dof,
                                                                                             loc, Sigma)

        return log_lik

    def sample_marginal(self, lenp, samples):
        """
        Sample the probability density h
        :return:
        """
        return self._sample_marginal(lenp, samples)

    def _sample_marginal(self, lenp, samples):
        """
        Sample the marginal likelihood h(x)
        :return:
        """

        xsample = np.zeros((samples, lenp))

        for p in range(lenp):
            loc = self._u0p
            Sigma = (self._bxp / self._axp) * ((self._cp + 1) / self._cp)
            dof = 2 * self._axp

            xsample[:, p] = scipy.stats.t.rvs(dof, loc=loc, scale=np.sqrt(Sigma), size=samples)

        return xsample

    def log_predictive_marginal(self, x, X, p=None):
        """
        Evaluate the log probability density log(h(x|X))
        :param x:    The input we wish to obtain the probability density for
        :param X:    The conditional input
        :param p:    The indices we wish to compute the probability density for.
        :return:

        Two cases: pass x_p and X_p, and calculate prod across p,
                   pass all x and all X, and a list of the relevant indexes to calculate across

        #TODO; calculate this in log-space to improve numerical stability
        """
        assert x.ndim == 2, 'NormalInverseGamma log_predictive_marginal(): Require 2d ndarray x'
        assert X.ndim == 2, 'NormalInverseGamma log_predictive_marginal(): Require 2d ndarray X'
        assert x.shape[1] == X.shape[1], 'NormalInverseGamma log_predictive_marginal(): {0} != {1}'.format(x.shape[1],
                                                                                                  X.shape[1])
        if p is not None:
            assert x.shape[1] >= len(p), 'NormalInverseGamma log_predictive_marginal(): {0} !> {1}'.format(x.shape[1], len(p))

        # Evaluate product of only desired indices
        if p is None:
            xp = np.asmatrix(x)
            Xp = np.asmatrix(X)
        elif isinstance(p, (list,)):
            xp = copy.deepcopy(np.asmatrix(x[:, p]))
            Xp = copy.deepcopy(np.asmatrix(X[:, p]))
        else:
            raise ValueError('NormalInverseGamma log_predictive_marginal(): p must be a list of indices or None.')
        assert xp.shape[1] == Xp.shape[1]
        assert np.isnan(xp).any() == False
        assert np.isnan(Xp).any() == False

        return np.log(self._pred_marginal(xp, Xp))

    def _pred_marginal(self, xi, xkmi):
        """
        Calculate the likelihood of xi conditioned on the data points in a cluster. For our choice of priors this is a
        product of t-distributions.

        :param xi:          the positions vector(s) whose conditional likelihood densities we want to calculate
            :type           2d-ndarray Ni * len(p)
        :param xkmi:        the conditioning variables belonging to the jth cluster (minus xn if that currently belongs in j)
            :type           2d-ndarray Nkmi * xdim
        :param p:           list of indices the xi values pertain to.
            :type           list
        :return:            the marginal conditional likelihood density for the input - h(xi|xkmi)
            :type           1d-ndarray Ni
        """

        nkmi = xkmi.shape[0]

        logxlik = 0
        for subp in range(xi.shape[1]):
            #logging.debug('gC._pred_marglikelihood_x: \tdimension {0} of {1}'.format(subp+1, xi.shape[1]))
            ckpmi = self._cp + nkmi
            axkpmi = self._axp + (nkmi/2.)
            ukpmi = (1./(self._cp + nkmi))*(self._cp*self._u0p + np.sum(xkmi[:, subp]))
            sumx = np.sum(np.power(xkmi[:, subp], 2))
            bxkpmi = self._bxp + (self._cp*self._u0p**2 - ckpmi*ukpmi**2+sumx)/2.

            loc = ukpmi
            Sigma = np.sqrt((bxkpmi/axkpmi)*((ckpmi+1)/ckpmi))
            dof = 2 * axkpmi

            logxlik += np.reshape(scipy.stats.t.logpdf(xi[:, subp], df=dof, loc=loc, scale=Sigma), (-1,))

        if (np.exp(logxlik) <= 0).all():
            logging.warning('NormalInverseGamma _pred_marginal() underflow h(x|X): {0}'.format(logxlik))

        return np.exp(logxlik)

    def sample_predictive_marginal(self, X, samples):
        """
        Sample h(x|X) where X.shape[1] determines P.

        :param X:
        :param samples:
        :return:
        """
        assert X.ndim == 2, 'NormalInverseGamma hX_sample() Require 2d array X, {0}'.format(X.ndim)

        return self._sample_pred_marginal(X, samples)

    def _sample_pred_marginal(self, xcond, samples):
        """Sample the predictive t-distribution"""

        xsample = np.zeros((samples, xcond.shape[1]))
        n = xcond.shape[0]

        # # New
        # for p in range(xcond.shape[1]):
        #     ckpmi = self._cp + n
        #     axkpmi = self._axp + (n / 2.)
        #     ukpmi = (1. / (self._cp + n)) * (self._cp * self._u0p + np.sum(xcond[:, p]))
        #     sumx = np.sum(np.power(xcond[:, p], 2))
        #     bxkpmi = self._bxp + (self._cp * self._u0p ** 2 - ckpmi * ukpmi ** 2 + sumx) / 2.
        #     loc = ukpmi
        #     Sigma = (bxkpmi / axkpmi) * ((ckpmi + 1) / ckpmi)
        #     dof = 2 * axkpmi
        #     xsample[:, p] = scipy.stats.t.rvs(df=dof, loc=loc, scale=Sigma, size=samples)

        # Old
        for p in range(xcond.shape[1]):
            ckpmi = self._cp + n
            axkpmi = self._axp + (n / 2.)
            ukpmi = (1. / (self._cp + n)) * (self._cp * self._u0p + np.sum(xcond[:, p]))
            sumx = np.sum(np.power(xcond[:, p], 2))
            bxkpmi = self._bxp + (self._cp * self._u0p ** 2 - ckpmi * ukpmi ** 2 + sumx) / 2.
            loc = ukpmi
            Sigma = np.sqrt((bxkpmi / axkpmi) * ((ckpmi + 1) / ckpmi))  # TODO: check rooting is correct
            dof = 2 * axkpmi
            xsample[:, p] = scipy.stats.t.rvs(df=dof, loc=loc, scale=Sigma, size=samples)

        return xsample

    # def plot_xlikelihood(self, x, covariate=0, path=None):
    #     """
    #     Plot the input likelihood and marginalised (conditioned on some example subsets) likelihood all in one figure.
    #     The subsets are shown as scatter points in the same colour. When input is multi dimensional, other covariates
    #     are fixed at their mean values. This leads to plotting a slice, which is important when interpreting results.
    #
    #     :param covariate:       The index of the covariate we wish to plot
    #         :type:              int
    #     """
    #     raise NotImplementedError
    #
    #     try:
    #         import matplotlib.pyplot as plt
    #     except:
    #         raise ImportError('cannot import matplotlib')
    #
    #     assert covariate <= x.shape[1], '_plot_xlikelihood: we do not have {0} covariates'.format(covariate)
    #
    #     # Test inputs
    #     test_points = 100
    #     Xtest = np.matlib.repmat(np.reshape(np.mean(x, axis=0), (1,-1)), test_points,1)
    #     Xtest[:, covariate] = np.linspace(np.min(x), np.max(x), test_points)
    #
    #     # evaluate marg likelihood and predictive marg input_models for different conditional cases
    #     mix = np.vstack((x[-5:, :], x[:5, :]))
    #     xmarglik = self.h(Xtest)
    #     xsmarglik = self.h_sample(1, 1000)
    #     xpredAll = self.hX(Xtest, x[:, :])
    #     xspredAll = self.hX_sample(x[:, :], 1000)
    #     xpredUpper = self.hX(Xtest, x[-5:, :])
    #     xspredUpper = self.hX_sample(x[-5:, :], 1000)
    #     xpredLower = self.hX(Xtest, x[:5, :])
    #     xspredLower = self.hX_sample(x[:5, :], 1000)
    #     xpredEnds = self.hX(Xtest, mix)
    #     xspredEnds = self.hX_sample(mix, 1000)
    #
    #     # plot the marginal likelihood
    #     plt.plot(Xtest[:, covariate], xmarglik, label='h(x)', color='y')
    #     plt.hist(xsmarglik, bins=30, density=True, color='y')
    #
    #     # plot the conditional marginal likelihood with different condition cases
    #     plt.plot(Xtest[:,covariate], xpredAll, label='h(x|black)', color ='k')
    #     plt.hist(xspredAll, bins=30, density=True, color='k')
    #     plt.scatter([x[:, covariate]], -0.0025*np.ones(x.shape[0]), c='k')
    #
    #     plt.plot(Xtest[:,covariate], xpredUpper, label='h(x|red)', color = 'r' )
    #     plt.hist(xspredUpper, bins=30, density=True, color='r')
    #     plt.scatter([x[-5:, covariate]], -0.005*np.ones(5), c='r')
    #
    #     plt.plot(Xtest[:,covariate], xpredLower, label='h(x|green)', color = 'g')
    #     plt.hist(xspredLower, bins=30, density=True, color='g')
    #     plt.scatter([x[:5, covariate]], -0.005 * np.ones(5), c='g')
    #
    #     plt.plot(Xtest[:,covariate], xpredEnds, label='h(x|blue)', color = 'b')
    #     plt.hist(xspredEnds, bins=30, density=True, color='b')
    #     plt.scatter([mix[:, covariate]], -0.0075 * np.ones(np.shape(mix)[0]), c='b')
    #
    #     # formatting plot
    #     plt.xlabel('x')
    #     plt.ylabel('density')
    #     plt.legend()
    #     plt.title('x input_models - covariate {0} with others fixed at mean'.format(covariate))
    #     if path != None:
    #         plt.savefig("".join([path, 'prior_x.png']))
    #     else:
    #         plt.show()
    #
    #     plt.clf()

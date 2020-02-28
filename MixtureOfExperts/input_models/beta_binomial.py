"""
Binomial local input model(s) for discrete ordered covariates taking values {0, 1,..., Gp}.

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

__all__ = ['BetaBinomial']


class BetaBinomial(InputModelBase):
    """
    A class for the Binomial local input model with conjugate Beta-Binomial prior distribution.
    """

    # public (accessible through @property decorators below)
    _Gp = None
    _gammap = None
    _domain = None

    @property
    def Gp(self):
        return self._Gp

    @property
    def gammap(self):
        return self._gammap

    def __init__(self, Gp, gammap, name='BetaBinomial'):
        """
        Initialise the BetaBinomial class

        :param Gp:                  The largest integer value the covariates take
        :param gammap:              Conjugate beta distribution parameters
        :param name:                Class name
        """
        assert (len(gammap) == 2), 'gamma_p must contain two elements'

        super(BetaBinomial, self).__init__(name=name)
        self._Gp = Gp
        self._domain = np.linspace(0, self.Gp, self.Gp + 1)
        self._gammap = gammap

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = "\nX LIKELIHOOD: " + self.__name__ + '\n'
        s += "\t Distribution over ordered inputs: {0} \n".format(str(self._domain))
        s += "\t With gamma_p: {0}".format(str(self.gammap))

        return s

    def log_marginal(self, x, p=None):
        """
        Evaluate the log probability density log(h(x))
        :param x:       input
        :param p:       covariates indices of x to calculate
        :return:

        Two cases: pass x_p and calculate prod across all,
                   pass all x and a list of the relevant indexes to calculate across
        """
        assert x.ndim == 2, 'BetaBinomial log_marginal(): x must be 2d array - {0}'.format(x.ndim)
        if p is not None:
            assert x.shape[1] >= len(p), 'BetaBinomial log_marginal(): covariates ({0},{1})'.format(x.shape[1], len(p))

        # Evaluate product of only desired indices
        if p is None:
            xp = np.asmatrix(x)
        elif isinstance(p, (list,)):
            xp = copy.deepcopy(np.asmatrix(x[:, p]))
        else:
            raise ValueError('BetaBinomial log_marginal(): p must be a list of indices.')
        assert np.isnan(xp).any() == False

        # Evaluate and return
        if x.shape[0] == 1:
            return self._log_marglikelihood_x_(xp)
        else:
            return self._log_joint_marglikelihood_x_(xp)

    def _log_marglikelihood_x_(self, x):
        """
        Calculate the marginal likelihood h(x).

        :param x:          The positions vector(s) whose likelihood densities we want to calculate
            :type          matrix [nj * P]
        :return:           The marginal likelihood density for the input [h(x_1),...,h(x_nj)]
            :type          vector [nj]
        """

        loghx = np.zeros((x.shape[0]))
        for i in range(x.shape[0]):
            loghx[i] = self._log_marglikelihood_x_single(np.reshape(x[i, :], (1, -1)))
        return loghx

    def _log_marglikelihood_x_single(self, xi):
        """
        Calculate the marginal likelihood of a single point across p, h(xi). Only used by _marglikelihood_x.

        :param xi:          the positions vector(s) whose likelihood densities we want to calculate
            :type           matrix [1 * xdim]
        :return:            the marginal likelihood density for the input - h(xi)
            :type           float
        """

        logxlik = 0
        for p in range(xi.shape[1]):
            if xi[0, p] in self._domain:
                logfraction1 = scipy.special.gammaln(np.sum(self.gammap)) -\
                               scipy.special.gammaln(self.gammap[0]) -\
                               scipy.special.gammaln(self._gammap[1])
                logfraction2 = scipy.special.gammaln(self.gammap[0] + xi[0, p]) +\
                               scipy.special.gammaln(self.gammap[1] + self.Gp - xi[0, p]) -\
                               scipy.special.gammaln(np.sum(self.gammap) + self.Gp)
                logfraction = logfraction1 + logfraction2
                loglikp = np.log(scipy.special.comb(self.Gp, xi[0, p])) + logfraction
                #logger.debug('_log_marglikelihood_x_single(): {0}'.format(loglikp))

                logxlik += loglikp
                assert np.isfinite(loglikp), 'BetaBinomial _log_marglikelihood_x_single(): underflow:' \
                                             ' {0},{1},{2}'.format(xi[0, p], logxlik, loglikp)
            else:
                #logging.warning('BB._marglikelihood_x_single(): ' +
                #                'x outside support, setting marginal likelihood equal to zero')
                logxlik += -np.inf

        return logxlik

    def _log_joint_marglikelihood_x_(self, X):
        raise NotImplementedError

    def sample_marginal(self, lenp, samples, returnbeta=False):
        """
        Sample the marginal likelihood h(x)
        :return:
        """
        xsample = np.zeros((samples, lenp))
        betasample = np.zeros((samples, lenp))

        for p in range(lenp):
            psip0 = self.gammap[0]
            psip1 = self.gammap[1]

            betasample[:, p] = np.random.beta(psip0, psip1, samples)
            for sample in range(samples):
                xsample[sample, p] = np.random.binomial(self.Gp, betasample[sample])

        if returnbeta:
            return xsample, betasample
        else:
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
        """
        assert x.ndim == 2, 'BetaBinomial log_predictive_marginal(): Require 2d ndarray x'
        assert X.ndim == 2, 'BetaBinomial log_predictive_marginal(): Require 2d ndarray X'
        assert x.shape[1] == X.shape[1], 'BetaBinomial log_predictive_marginal(): P ({0},{1})'.format(x.shape[1],
                                                                                                      X.shape[1])
        if p is not None:
            assert x.shape[1] >= len(p), 'BetaBinomial log_predictive_marginal(): P ({0},{1})'.format(x.shape[1],
                                                                                                      len(p))
        assert all(conditionals in self._domain for conditionals in X)

        # Evaluate product of only desired indices
        if p is None:
            xp = np.asmatrix(x)
            Xp = np.asmatrix(X)
        elif isinstance(p, (list,)):
            xp = copy.deepcopy(np.asmatrix(x[:, p]))
            Xp = copy.deepcopy(np.asmatrix(X[:, p]))
        else:
            raise ValueError('BetaBinomial log_predictive_marginal(): p must be a list of indices or None.')
        assert xp.shape[1] == Xp.shape[1]
        assert np.isnan(xp).any() == False
        assert np.isnan(Xp).any() == False

        return self._log_pred_marglikelihood_x_(xp, Xp)

    def _log_pred_marglikelihood_x_(self, xi, xkmi):
        """
        Calculate the likelihood of xi conditioned on the data points in a cluster.

        :param xi:          the positions vector(s) whose conditional likelihood densities we want to calculate
            :type           2d-ndarray Ni * xdim
        :param xkmi:        the conditioning variables belonging to the jth cluster (minus xn if that currently belongs in j)
            :type           2d-ndarray Nkmi * xdim
        :return:            the marginal conditional likelihood density for the input - h(xi|xkmi)
            :type           1d-ndarray Ni
        """

        loghxiXkmi = np.zeros((xi.shape[0]))
        for i in range(xi.shape[0]):
            loghxiXkmi[i] = self._log_pred_marglikelihood_x_single(np.reshape(xi[i, :], (1, -1)), xkmi)
        return loghxiXkmi

    def _log_pred_marglikelihood_x_single(self, xi, xkmi):
        """
        Calculate the likelihood of xi conditioned on the data points in a cluster.
        Only used by _pred_marglikelihood_x - TODO: instead vectorise that.

        :param xi:          the positions vector(s) whose conditional likelihood densities we want to calculate
            :type           2d-ndarray 1 * xdim
        :param xkmi:        the conditioning variables belonging to the jth cluster (minus xn if that currently belongs
                            in j)
            :type           2d-ndarray Nkmi * xdim
        :return:            the marginal conditional likelihood density for the input - h(xi|xkmi)
            :type           float
        """

        nkmi = xkmi.shape[0]
        #xlik = 1
        logxlik = 0
        for p in range(xi.shape[1]):
            if xi[0, p] in self._domain:
                psihatp0k = self.gammap[0] + nkmi * np.mean(xkmi[:, p])
                psihatp1k = self.gammap[1] + nkmi * (self.Gp - np.mean(xkmi[:, p]))
                logfraction1 = scipy.special.gammaln(np.sum(self.gammap) + self.Gp * nkmi) - \
                               scipy.special.gammaln(psihatp0k) - \
                               scipy.special.gammaln(psihatp1k)
                logfraction2 = scipy.special.gammaln(psihatp0k + xi[0, p]) + \
                               scipy.special.gammaln(psihatp1k + self.Gp - xi[0, p]) - \
                               scipy.special.gammaln(np.sum(self.gammap) + (self.Gp * (nkmi + 1)))
                logfraction = logfraction1 + logfraction2
                #likp = np.exp(np.log(scipy.special.comb(self.Gp, xi[0, p])) + logfraction)
                loglikp = np.log(scipy.special.comb(self.Gp, xi[0, p])) + logfraction
                #logger.debug('_log_pred_marglikelihood_x_single(): {0}'.format(loglikp))

                # xlik *= likp
                #assert likp > 0, 'underflow in input predictive marginal likelihood {0},{1},{2}'.format(xi[0, p], xlik, likp)

                logxlik += loglikp
                assert np.isfinite(loglikp), '_log_pred_marglikelihood_x_single(): underflow {0},{1},{2}'.format(xi[0, p], logxlik, loglikp)
            else:
                #logging.warning('BB._marglikelihood_x_single(): ' +
                #                'x outside support, setting marginal likelihood equal to zero')
                #xlik *= 0
                logxlik += -np.inf

        return logxlik

    def sample_predictive_marginal(self, X, samples, returnbeta=False):
        """
        Sample h(x|X) where X.shape[1] determines P.

        :param X:
        :param samples:
        :return:
        """
        assert X.ndim == 2, 'BetaBinomial sample_predictive_marginal(): Require 2d ndarray X'

        xsample = np.zeros((samples, X.shape[1]))
        betasample = np.zeros((samples, X.shape[1]))

        n = X.shape[0]
        for p in range(X.shape[1]):
            psihatp0k = self.gammap[0] + n * np.mean(X[:, p])
            psihatp1k = self.gammap[1] + n * (self.Gp - np.mean(X[:, p]))

            betasample[:, p] = np.random.beta(psihatp0k, psihatp1k, samples)
            for sample in range(samples):
                xsample[sample, p] = np.random.binomial(self.Gp, betasample[sample])

        if returnbeta:
            return xsample, betasample
        else:
            return xsample

    def plot_xlikelihood(self, x, covariate=0, path=None):
        """
        Plot the input likelihood and marginalised (conditioned on some example subsets) likelihood all in one figure.
        The subsets are shown as scatter points in the same colour. When input is multi dimensional, other covariates
        are fixed at their mean values. This leads to plotting a slice, which is important when interpreting results.

        :param covariate:       The index of the covariate we wish to plot
            :type:              int
        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError('cannot import matplotlib')

        assert covariate <= x.shape[1], '_plot_xlikelihood: we do not have {0} covariates'.format(covariate)

        # Test inputs
        test_points = self.Gp + 1
        Xtest = np.floor((self.Gp + 1)/2) * np.ones((test_points, x.shape[1]))
        Xtest[:, covariate] = np.linspace(0,self.Gp,self.Gp+1)
        n = x.shape[0]

        # evaluate marg likelihood and predictive marg input_models for different conditional cases
        mix = np.vstack((x[-5:, :], x[:5, :]))
        xmarglik = self.h(Xtest)
        xsmarglik = self.h_sample(1, 1000)
        xpredAll = self.hX(Xtest, x[:, :])
        xspredAll = self.hX_sample(x[:, :], 1000)
        xpredUpper = self.hX(Xtest, x[-5:, :])
        xspredUpper = self.hX_sample(x[-5:, :], 1000)
        xpredLower = self.hX(Xtest, x[:5, :])
        xspredLower = self.hX_sample(x[:5, :], 1000)
        xpredEnds = self.hX(Xtest, mix)
        xspredEnds = self.hX_sample(mix, 1000)

        # plot
        fig = plt.figure()
        ax = plt.subplot(111)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # plot the marginal likelihood
        ax.plot(Xtest[:, covariate], xmarglik, 'y.-', label='h(x)')
        ax.hist(xsmarglik, bins=50, weights=np.ones_like(xsmarglik) / len(xsmarglik), color='y')

        # plot the conditional marginal likelihood with different condition cases
        ax.plot(Xtest[:, covariate], xpredAll, 'k.--', label='h(x|black)')
        ax.hist(xspredAll, bins=30, weights=np.ones_like(xspredAll)/len(xspredAll), color='k')
        ax.scatter([x[:, covariate]], -0.025 * np.ones(n), c='k')

        ax.plot(Xtest[:, covariate], xpredEnds, 'b.:', label='h(x|blue)', )
        ax.hist(xspredEnds, weights=np.ones_like(xspredEnds)/len(xspredEnds), bins=30, color='b')
        ax.scatter([mix[:, covariate]], -0.075 * np.ones(np.shape(mix)[0]), c='b')

        ax.plot(Xtest[:, covariate], xpredUpper, 'r.--', label='h(x|red)')
        ax.hist(xspredUpper, weights=np.ones_like(xspredUpper)/len(xspredUpper), bins=30, color='r')
        ax.scatter([x[-5:, covariate]], -0.05 * np.ones(5), c='r')

        ax.plot(Xtest[:, covariate], xpredLower, 'g.--', label='h(x|green)')
        ax.hist(xspredLower, weights=np.ones_like(xspredLower)/len(xspredLower), bins=30, color='g')
        ax.scatter([x[:5, covariate]], -0.05 * np.ones(5), c='g')

        # formatting plot
        plt.xlabel('x')
        plt.ylabel('density')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('x input_models - covariate {0} with others fixed at median'.format(covariate))
        # save or show plot
        if path != None:
            plt.savefig("".join([path, 'prior_x.png']))
        else:
            plt.show()

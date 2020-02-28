"""
A class for link functions. These are link functions mapping responses to latent
responses on the real line.

TODO: put all probit models into one class with inheritance.

Author:
    Charles Gadd

Date:
"""
from __future__ import division
from MixtureOfExperts.utils import tmvtnorm
import numpy as np
from scipy.stats import norm
import logging
logger = logging.getLogger(__name__)

__all__ = ['orderedProbit']


class orderedProbit(object):
    """
    A class for the ordered probit model for ordinal outputs.
    """

    # public (accessible through @property decorators below)

    # private
    _expert = None              # expert
    _L = None                   # Number of (linearly spaced) blocks (outputs = 0, 1, ..., L-1)
    _epsilon0 = None            # Start of linear spacing
    _epsilonLm1 = None          # end of linear spacing
    _epsilon = None             # the edges

    @property
    def categories(self):
        """Return the discrete categories for the probit model"""
        #print(np.hstack((self._epsilon[1:-1], self._L)))
        return np.hstack((self._epsilon[1:-1], self._L))

    def __init__(self, expert, L, minmax, name='ordinal probit'):
        """
        Initialize the class.

        :param expert:          # expert
        """
        self.__name__ = name
        self._expert = expert
        self._L = L
        self._epsilon0 = minmax[0]
        self._epsilonLm1 = minmax[1]

        epsilon = np.linspace(self._epsilon0, self._epsilonLm1, self._L)
        epsilon = np.insert(epsilon, np.shape(epsilon), np.inf)
        self._epsilon = np.insert(epsilon, 0, -np.inf)

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = "\nLINK FUNCTION: " + self.__name__
        s += "\n\t epsilon: {0} \n".format(str(self._epsilon))
        s += "\n\t L: {0}".format(str(self._L))
        s += "\n\t categories: {0}".format(str(self.categories))
        return s

    def sample_latent_response(self, X, Z, Y, theta, samples=1, position=None):
        """
        :param X             Covariates
        :param Z             Feature cluster allocations
        :param Y             Raw features [0,1,2,...,Nunique-1]
        :param theta:        Matrix containing hyper-parameters for all current clusters.
        :param samples:      Number of latent response samples to draw for each datum, default 1.
        :return:             Latent samples (binary)
        """
        Ylatent = np.zeros_like(Y)
        for j in np.unique(Z):
            thetaJ = theta[int(j), :]                                                       # Hyper-params of cluster j
            xj = X[Z == j]                                                                  # Get which pairs are in j
            yj = Y[Z == j]
            Nj = np.shape(xj)[0]                                                            # Size of cluster j

            for feature in range(Y.shape[1]):
                itershift = feature * self._expert._n_hyper_indGP                           # for adding hypers back

                # Expert mean
                if self._expert._process_mean is not False:
                    mean_function = self._expert.process_mean

                    if self._expert._process_mean == 'Linear':
                        mean_function.A = thetaJ[self._expert.index_mean + itershift, None]
                        logging.debug('Linear mean function. Setting A when sampling probit model.')
                    elif self._expert._process_mean == 'Constant':
                        mean_function.C = thetaJ[self._expert.index_mean + itershift, None]
                        logging.debug('Constant mean function. Setting C when sampling probit model.')
                    else:
                        raise ValueError('Bad mean function')

                    mean = mean_function.f(xj)[:, 0]
                else:
                    mean = np.zeros((xj.shape[0], 1))[:, 0]

                # Expert covariance
                kernel = self._expert.kern
                kernel.variance = thetaJ[self._expert.index_signal + itershift]                 # Add hyper-parameters
                kernel.lengthscale = thetaJ[self._expert.index_lengthscale + itershift]         # Add hyper-parameters
                covariance = kernel.K(xj, xj) + (thetaJ[self._expert.index_nugget] * np.eye(xj.shape[0]))

                # Sample latent response from truncated multivariate normal, if yji == 0 (-inf, 0)
                lower = [self._epsilon[int(yj[i])] for i in range(Nj)]
                upper = [self._epsilon[int(yj[i])+1] for i in range(Nj)]

                if position is not None:
                    pos0 = position[Z==j]
                else:
                    pos0 = None

                try:
                    Ylatent[Z == j, feature] = tmvtnorm.tmvtnorm_sample(samples, mean, covariance, lower, upper,
                                                                        position=pos0)
                    assert (np.all(~np.isinf(Ylatent))), 'Bad latent Y for features allocated to cluster {0}'.format(j)
                    assert (np.all(~np.isnan(Ylatent))), 'Bad latent Y for features allocated to cluster {0}'.format(j)
                except:
                    logging.exception(thetaJ)
                    logging.exception(xj[0:1, :])
                    logging.exception(np.ndarray.flatten(covariance))
                    logging.exception(np.ndarray.flatten(kernel.K(xj, xj)))
                    raise ValueError

        return Ylatent

    def project_point(self, latpoints):
        """
        Project points on the latent response back into feature space
        """
        if latpoints.ndim == 1:
            latpoints = np.reshape(latpoints, (-1, 1))

        N = latpoints.shape[0]          # number of points to project
        D = latpoints.shape[1]          # dimensionality of response
        assert D == 1, 'Not generalised to multi-output'

        Y = np.zeros((N, 1))
        for n in range(N):
            for l in range(len(self.categories)):           # for readability we do using a loop, can be done quicker.
                if latpoints[n, 0] > self._epsilon[l] and latpoints[n, 0] < self._epsilon[l+1]:
                    #logging.debug('{0} between ({1},{2}), l:{3}'.format(latpoints[n, 0], self._epsilon[l], self._epsilon[l+1],l))
                    Y[n, :] = l
                    break

        return Y

    def project_density(self, latmeans, latstd):
        """
        Project density on the latent response back into feature space. From Gaussian moments
        """
        assert np.shape(latmeans) == np.shape(latstd)
        assert latmeans.ndim == latstd.ndim == 1

        Ncovar = latmeans.shape[0]

        ydens = np.zeros((len(self.categories), Ncovar))
        for i in range(Ncovar):
            for l in range(len(self.categories)):
                #print('bounds ({0}, {1})'.format(self._epsilon[l], self._epsilon[l+1]))
                ydens[l, i] = norm.cdf((self._epsilon[l+1] - latmeans[i]) / latstd[i]) - \
                              norm.cdf((self._epsilon[l] - latmeans[i]) / latstd[i])
                assert ~np.isnan(ydens[l, i]), 'probit model covariate {0} density {1} = {2}'.format(l, i, ydens[l, i])

            assert np.isclose(np.sum(ydens[:, i]), 1), 'sum = {0}'.format(np.sum(ydens[:, i]))

        #import matplotlib.pyplot as plt
        #xstar = np.linspace(-10, 50, 100)
        #plt.plot(xstar, norm.pdf(xstar, loc=latmeans, scale=latvars))
        #plt.scatter(self.categories, ydens[:, 0])
        #plt.show()

        return ydens

    def project_pred_density(self, lat_pred_means, lat_pred_std):
        #logger.warning('Use project_density()')
        return self.project_density(lat_pred_means, lat_pred_std)

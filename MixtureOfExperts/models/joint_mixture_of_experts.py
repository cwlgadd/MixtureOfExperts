"""
A class for the Joint Mixture of Experts model.

Author:
    Charles Gadd

Date:
    09/10/2017
"""
from __future__ import division
import itertools, inspect, sys, os
from MixtureOfExperts.utils import progress_bar as pb
import scipy.stats
import numpy as np
import random
import copy
import pickle
import logging
logger = logging.getLogger(__name__)
import time

__all__ = ['JointMixtureOfExperts']

class JointMixtureOfExperts(object):
    """
    A class for the Joint Mixture of Experts model.
    """

    # public (accessible through @property below)
    _X = None                          # NxD ndarray of covariates (missing data _not_ allowed)
    _Y = None                          # NxD ndarrray of features (missing data _not_ allowed)
    _alpha = None                      # concentration parameter
    _MCMCstates = None                 # the chain of states from our algorithm. list of dictionaries

    # private
    _expert = None                     # The object used to model the experts. Must inherit certain functions
    _xlikelihood = None                # The likelihood on the inputs x
    _probit = None                     # optional probit model for ordinal/binary responses
    _latenty = None                    # latent y, used only when we have a probit model
    _mcmcSteps = None                  # number of Monte Carlo iterations that have been performed
    _mNeal8 = None                     # number of new clusters each reallocation step
    _k = None                          # current number of clusters
    _sIndex = None                     # vector of cluster allocation indexes for each data point.
    _theta = None                      # matrix containing hyper-parameters, updated rather than appended.
    _ua = None
    _va = None

    @property
    def x(self):
        """
        Get the covariates.
        """
        return self._X

    @property
    def xdim(self):
        """
        Get the number of covariate dimensions.
        """
        return self._X.shape[1]

    @property
    def n(self):
        """
        Get the number of data points.
        """
        return self._X.shape[0]

    @property
    def y(self):
        """
        Get the features
        """
        if self._probit is None:                            # if we have no probit model work on raw features
            return self._Y
        elif self._latenty is not None:                     # if we have probit and latent available return latent
            return self._latenty
        else:                                               # if we have probit but latent not available return error
            # this should only ever be reached by calling decorator before optimising.
            raise ValueError('We have a probit model, but have not calculated the latent response yet.')

    @property
    def ydim(self):
        """
        Get the number of feature dimensions
        """
        return self._Y.shape[1]

    @property
    def alpha(self):
        """
        Get the concentration parameter alpha
        """
        return self._alpha

    @property
    def k_unique(self):
        raise NotImplementedError

    @property
    def state(self):
        """
        Get the current state of our class
        """
        if self._mcmcSteps == 0:
            current_state = {'k_counter': self._k,
                             'sIndex': self._sIndex,
                             'theta': self._theta,
                             'alpha': self._alpha,
                             'latent': self._Y}
        else:
            current_state = {'k_counter': self._k,
                             'sIndex': self._sIndex,
                             'theta': self._theta,
                             'alpha': self._alpha,
                             'latent': self.y}

        self.check_consistent(state=current_state)

        return current_state

    @property
    def states(self):
        """
        Get the chain of states of our algorithm
        """
        return self._MCMCstates

    @property
    def indexes(self):
        """
        Get the chain of assigned indexes from our states
        """
        if self._mcmcSteps > 0:
            indexMatrix_feature = np.zeros((self._mcmcSteps, self.n))

            for sample in range(self._mcmcSteps):
                state = self.states[sample]['sIndex']
                indexMatrix_feature[sample, :] = state[:, 0]
        else:
            indexMatrix_feature = self._sIndex[:, 0]

        return indexMatrix_feature

    def __init__(self, x, y, expert, xlikelihood, probit=None, mneal8=4, initialisation=6, name='JmE'):
        """
        Initialize the model.

        :param x:               Observed features
            :type               np.ndarray (#samples * #features)
        :param y:               Observed covariates
            :type               np.ndarray (#samples * #covariates)
        """

        self.__name__ = name

        try:
            self.load()
            logger.info('Loaded Joint Mixture of Experts model from root directory {0}, continuing'.format(os.getcwd()))
        except:
            logger.debug('Creating new Joint Mixture of Experts model')
            if x.ndim == 1:
                x = np.reshape(x, (-1,1))
            if y.ndim == 1:
                y = np.reshape(y, (-1,1))

            assert x.ndim == 2 and y.ndim == 2, '_init: require 2d ndarrays. ({0} & {1})!=2'.format(x.ndim, y.ndim)
            assert x.shape[0] == y.shape[0], '_init: inconsistent sample size. {0} != {1}'.format(x.shape[0], y.shape[0])
            assert mneal8 >= 1, '_init: must have at least one proposed new cluster. {0}!>=0'.format(mneal8)

            """ model """
            self._X = x
            self._Y = y
            self._expert = expert
            self._xlikelihood = xlikelihood
            self._probit = probit

            # Monte Carlo parameterisations
            self._mcmcSteps = 0
            self._MCMCstates = []
            self._mNeal8 = mneal8

            # Memory allocation, we have at most n clusters, +1 is for overflow new if in extreme case of n singletons.
            self._theta = np.zeros((self.n + 1, self._expert.n_hyper))  # GP hyper-parameter
            self._sIndex = np.zeros((self.n, 1))                               # Allocation index, col1=feature, col2=input

            """ prior for concentration parameter alpha """
            self._ua = 1
            self._va = 1

            """ initialise clusters """
            if initialisation == 'singleton':         # start with all points initially allocated to own cluster
                self._sIndex[:, 0] = np.arange(self.n)
            elif isinstance(initialisation, int):     # sample allocations uniformly from np.arange(intialisation)
                self._sIndex[:, 0] = np.random.choice(initialisation, self.n)
            elif len(initialisation) == self.n:       # allocate to specified clusters
                self._sIndex[:, 0] = initialisation
            else:
                logging.critical('Exiting. Unable to intialise allocations. initialisation: {0}'.format(initialisation))
                sys.exit()
            self._k = int(np.max(np.unique(self._sIndex[:, 0])) + 1)  # number of unique clusters

            """ set parameters describing intitial state by sampling posteriors & priors """
            self._alpha = 2                       # initial concentration parameter
            rows = np.asarray(np.unique(self._sIndex[:, 0]), dtype=np.int16)  # which feature clusters are used for sampling
            self._theta[rows, :] = self._expert.prior(samples=len(np.unique(self._sIndex[:, 0])))

            """ if we are using a probit model, start with a high GP variance so we can sample an initial latent y"""
            if self._probit is not None and len(self._expert.index_nugget) > 0:
                # assert self._expert._constrain_nugget is False
                for feature in range(self.ydim):
                    itershift = feature * self._expert._n_hyper_indGP
                    self._theta[rows, self._expert.index_nugget + itershift] = 1 * np.std(self._Y[:, feature])

            """ add this first state to chain """
            self._append_state()                    # initial state at 0 index

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = "\nMODEL, JmE: " + self.__name__ + '\n'
        s += "\t Number of training points: {0} \n".format(str(self.n) )
        s += "\t Number of mNeal8: {0} \n".format(str(self._mNeal8))
        s += "\t Number of MCMC steps completed: {0} \n".format(str(self._mcmcSteps))
        s += str(self._expert)
        s += str(self._xlikelihood)
        if self._probit is not None:
            s += str(self._probit)

        return s

    def __call__(self, samples=250):
        """
        Run algorithm for posterior exploration of the model.
        """
        times = np.zeros(samples,)

        for i in range(samples):  #itertools.repeat(None, samples):
            logger.info("Iteration {0}".format(self._mcmcSteps+1))
            t_start = time.time()

            # non-continuous output through data augmentation, sample latent y
            if self._probit is not None:
                logger.info('Sampling probit model')
                self._latenty = self._probit.sample_latent_response(self._X, self._sIndex[:, 0], self._Y, self._theta,
                                                                    position=self._latenty)
            self._remove_empty_()
            self.check_consistent('call(): remove empty')

            # algorithm
            self._update_allocations()
            self._update_hyperparameters()
            self._update_concentration()

            # save new step
            self._append_state()
            self._mcmcSteps += 1
            self.check_consistent('call(): after update')

            times[i] = time.time() - t_start

        return times

    def save(self, root=''):
        """
        Save class as 'self.__name__'.xml. Includes classes defined in the hierarchy, such as expert.
        """
        with open(root + self.__name__ + '.xml', 'wb') as selfFile:
            pickle.dump(self.__dict__, selfFile, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug('Saved model')

    def load(self, root=''):
        """
        Load class from 'self.__name__'.xml
        """
        with open(root + self.__name__ +'.xml', 'rb') as selfFile:
            self.__dict__.update(pickle.load(selfFile))
            logger.debug('Loaded model')

    def _update_hyperparameters(self):
        """
        update the cluster hyper parameters by sampling the posterior using Hamiltonian Monte Carlo
        """
        logger.info("updating hyper parameters for {0} unique cluster(s), K={1}".format(len(np.unique(self._sIndex)),
                                                                                      self._k))

        for k in range(self._k):                                 # for each feature cluster.
            if sum(self._sIndex[:, 0] == k) == 0:                # if empty don't update and remove from previous.
                self._theta[k, :] = np.zeros((self._theta.shape[1]))
            else:                                                # else update, initialising at last value
                theta_old = copy.copy(self._theta[k, :])
                self._theta[k, :], _ = self._expert.posterior(self.x[self._sIndex[:, 0] == k, :],
                                                              self.y[self._sIndex[:, 0] == k, :], self._theta[k, :],
                                                              samples=15)
                assert np.all(self._theta[k, :] != theta_old), \
                    'MC did not mix: cluster {0}, samples:\n {1} \n{2}'.format(k, theta_old, self._theta[k, :])
                logging.debug('hyper-parameters for cluster {0}: {1}'.format(k, self._theta[k, :]))

        self.check_consistent('update hyper')

    def _update_allocations(self):
        """
        update allocation variables via a Polya Urn process.

        :start:      class consists of a set of clusters (none empty), each with new sampled hyper-parameters
        :process:    update clusters by re-allocating each point. Empty clusters are deleted within the process.
        :end:        class with updated clusters (none empty) If a new cluster has been made the hyper-parameters are
                    draws from the priors, which should be resampled before calling this function.
        """
        start_time = time.time()
        logger.info("Updating allocations")

        if self._mcmcSteps < 20:                       # mix the order in initial iters to avoid cluster size imbalance.
            order = random.sample(range(self.n), self.n)
        else:
            order = range(self.n)

        for i in order:                                                     # Update allocations for each data point
            logger.debug("\t sample {0} with latent response {1} target {2}".format(i, self.y[i, :], self._Y[i, :]))

            xi = np.reshape(self.x[i, :], (1, -1))
            yi = np.reshape(self.y[i, :], (1, -1))
            currentIndex = self._sIndex[i, 0]                               # To check if cluster empty at end
            # Memory allocaiton
            logap = -np.inf * np.ones(self._k + self._mNeal8)               # log alloc prob, memory alloc

            """ allocation probability for each occupied cluster (excluding sample i) """
            for j in range(self._k):    # calculate the allocation probabilities for each cluster.

                sjIndex = (self._sIndex[:, 0] == j)     # boolean for samples in jth feature cluster
                sjmiIndex = np.delete(sjIndex, i, 0)    # as above, minus current sample i
                nkmi = sum(sjmiIndex)                   # #samples in f cluster j, excluding i.

                if nkmi == 0:
                    pass                    # if singleton cluster, don't allocate back into its own cluster.
                                            # Instead, we use this cluster as one of the mNeal8 proposed clusters.
                else:
                    # get feature cluster datum, minus xi -> Xk-i and Yk-i
                    Xkmi = np.delete(self.x, i, 0)[sjmiIndex]
                    Ykmi = np.delete(self.y, i, 0)[sjmiIndex]

                    # Expert likelihood - GP predictive likelihood of data vector i
                    gploglikxi = self._expert.log_pred_marg_likelihood(Xkmi, Ykmi, xi, yi, self._theta[j, :])

                    # allocation prob
                    logap[j] = np.log(nkmi) + gploglikxi + self._xlikelihood.log_predictive_marginal(xi, Xkmi)

                    if not np.isfinite(logap[j]):
                        logging.warning('acceptance probability is not finite: {0}'.format(logap[j]))

            """ Allocation probability for mNeal8 new (and current if singleton) y-clusters by sampling priors """
            # sample new clusters
            new_theta = self._expert.prior(self._mNeal8)                             # samples * n_hyper
            if np.sum(self._sIndex[:, 0] == self._sIndex[i, 0]) == 1:                # replace first with singleton
                new_theta[0, :] = self._theta[int(self._sIndex[i, 0]), :]
            # allocation probability for new cluster (and existing cluster if a singleton).
            for im in range(self._mNeal8):                                             # allocation probability
                logap[self._k + im] = np.log(float(self._alpha)) - \
                                      np.log(self._mNeal8) + \
                                      self._expert.log_marg_likelihood(new_theta[im, :], xi, yi) + \
                                      self._xlikelihood.log_marginal(xi)

                assert np.isfinite(logap[int(self._k) + im]), \
                    'log acceptance probability is not finite: {0}'.format(logap[int(self._k) + im])

            """ exp-normalize trick """
            logap -= np.max(logap[logap != -np.inf])

            """ combine all allocation probabilities and normalise """
            apnorm = np.exp(logap) / sum(np.exp(logap))
            logging.debug('\t apnorm above 0.1: {0}'.format(apnorm[apnorm > 0.1]))

            try:
                sumapnorm = sum(apnorm)
                assert np.isclose(sumapnorm, 1), "alloc prob does not sum to one: {0}".format(sum(sum(apnorm)))
            except:
                logging.critical(logap)
                logging.critical(apnorm)
                logging.critical([type(iter) for iter in apnorm])
                logging.critical('Unable to sum over normalised probabilities. Ensure they are finite numbers.'
                                 ' \n {0} \n {1}'.format(logap, apnorm))
                raise SystemExit(1)

            apcum = np.cumsum(np.ndarray.flatten(apnorm))

            """ Allocate point using and update cluster parameters """
            apbool = scipy.stats.uniform(0, 1).rvs(1) > apcum
            idx_alloc = np.where(apbool == 0)[0][0]                   # index over existing clusters and m (or m-1) new

            if idx_alloc < self._k:         # allocate xi to an existing cluster
                logger.debug("\t ... to an existing cluster, index #{0}".format(idx_alloc))

                self._sIndex[i, 0] = idx_alloc
                # theta and self._k do not change

            else:                           # create the new cluster then delete empty clusters if necessary
                logger.debug("\t ... to an mNeal8 proposal, proposed index #{0}".format(idx_alloc - self._k))
                if np.sum(self._sIndex[:, 0] == self._sIndex[i, 0]) == 1 and idx_alloc == self._k:      # if singleton cluster
                    logger.debug("\t\t ... to the same singleton cluster")

                # add new cluster and update allocation for xi
                self._sIndex[i, 0] = self._k
                self._theta[self._k, :] = new_theta[idx_alloc - self._k, :]
                self._k += 1


            """ care taking - remove parameterisation for emptied clusters """
            if sum(self._sIndex[:, 0] == currentIndex) == 0:  # original feature cluster emptied
                self._theta[int(currentIndex), :] = np.zeros((self._theta.shape[1]))

            """ care taking - if full remove empty - emptied cluster index's are removed and k counters adjusted """
            if self._k > self.n:
                logger.info("\t\t\t ... removing empty clusters")
                self._remove_empty_()

        # assertions
        self.check_consistent('update allocations')

        logger.debug(f'Update allocations took {time.time() - start_time} seconds')

    def _update_concentration(self):
        """
        update concentration parameter via auxiliary variable technique - cite Bayesian density estimation and inference
        using mixtures.
        """
        currentAlpha = self._alpha

        phi = np.random.beta(currentAlpha+1, self.n)

        vahat = self._va - np.log(phi)

        uahat = self._ua + len(np.unique(self._sIndex))
        probability = (self.n*vahat) / (self.n*vahat + self._ua + len(np.unique(self._sIndex)) - 1)
        if np.random.uniform(0, 1) < probability:
            uahat = uahat - 1

        self._alpha = np.random.gamma(uahat, 1./vahat)
        logger.debug('concentration update: alpha {0}->{1}, prob {2}'.format(currentAlpha, self._alpha, probability))

        #plt.hist(np.random.gamma(uahat, vahat, 1000), 30, normed=True)
        #plt.title('alpha \sim Gamma({0},{1})'.format(uahat,vahat))
        #plt.show()

    def _remove_empty_(self):
        """
        Remove empty clusters
        """

        # initialise a copy
        effK = 0
        new_theta = np.zeros_like(self._theta)
        new_sIndex = -np.ones_like(self._sIndex)

        for k in range(self._k):
            # for each cluster
            nk = np.sum(self._sIndex == k)  # number of points in that cluster.

            if nk > 0:
                # if not empty
                new_sIndex[self._sIndex == k] = effK
                new_theta[effK, :] = self._theta[k, :]
                effK += 1

        self._k = effK
        self._sIndex = new_sIndex
        self._theta = new_theta

        assert (self._theta[self._k:, :] == 0).all()                 # check unused clusters are emptied

    def _append_state(self):
        """
        Add current state of the class to the chain of states as a list of dictionaries.
        """

        self.check_consistent(state=self.state)
        self._MCMCstates.append(copy.deepcopy(self.state))

    def check_consistent(self, last_update='', state=None):
        """
        Check consistency of model indexing. We assume that sIndex is correctly assigned throughout and compare other
        parameters to this as a benchmark.
        """
        if state is None:  # check consistency of current state of the model
            state = self.state

        k_counter = state['k_counter']
        sIndex = state['sIndex']
        theta = state['theta']
        alpha = state['alpha']

        #assert self._k <= self.n

        # counters
        assert k_counter >= max(sIndex[:, 0])

        # mass parameter for feature clusters
        assert alpha > 0

        rows = np.asarray(np.unique(sIndex[:, 0]), dtype=np.int16)

        # hyper-parameters
        theta_filled = theta[rows, :]
        theta_empty = np.delete(theta, rows, axis=0)
        assert np.any(theta_filled != 0), \
            'check_consistent: ' + last_update + ': theta_filled not filled: {0}'.format(theta_filled)
        assert np.all(theta_empty == 0), \
            'check_consistent: ' + last_update + ': feature_empty not empty: {0}'.format(theta_empty)

    def _predict_expectation(self, x_test, MC=None, burnin=0, thinning=1, return_latent=False):
        """
        Given a test point(s), find the expectation of the response across all states after burnin.

        :param x_test:      Input matrix for which we find the density of a given feature, averaged over all states.
            :type           2d-ndarray n_test * covariate dimension
        :param burnin       number of Markov Chain Monte Carlo samples discarded for the burn-in period.
            :type           int

        :return:           the mean response of the EmE model at x_test over Markov samples burnin to MC.
            :type
        """
        if MC is None:
            MC = self._mcmcSteps
        else:
            assert MC <= self._mcmcSteps, '_predict_density: MC={0} > completed {1}'.format(MC, self._mcmcSteps)

        assert len(self.states) > 1, 'JmE._pred_expec: Need to update model at least once to sample GP hyper'
        assert MC > burnin, 'JmE._pred_expec: burnin must be less than #MCMC samples. {0}!>{1}'.format(MC, burnin)
        assert x_test.ndim == 2, 'JmE._pred_expec: Require 2d ndarray input x_test'
        assert x_test.shape[1] == self.xdim, 'JmE._pred_expec: Test covariate dim must be equal to training dim'

        # Memory allocation
        C = np.zeros((x_test.shape[0]))                                          # Normalising constant
        if self._probit is None or return_latent is True:                        # Expectation (un-normalised)
            E = [np.zeros(x_test.shape[0]) for _ in range(self.ydim)]
        if self._probit is not None:
            E_projected = [np.zeros(x_test.shape[0]) for _ in range(self.ydim)]  # Expectation projected

        # Pre-computations
        hxs = np.exp(self._xlikelihood._log_full_marginal(x_test))                   # Marginal likelihood

        for m in range(MC - burnin):
            pb.update_progress((m+1)/(MC - burnin), message='JmE._predict_expectation()')
            if np.mod(m, thinning):
                logger.debug('JmE._pred_expec: Thinning by skipping m: {0}'.format(m + burnin))
                continue

            # Load state
            stateM = self.states[m + burnin]
            am = stateM['alpha']
            thetam = stateM['theta']
            sIndexm = stateM['sIndex']
            km = stateM['k_counter']

            # Prior weight
            prior_weight = (am * hxs) / (am + self.n)
            C += prior_weight
            if self._probit is None or return_latent is True:
                E = [E[i] + prior_weight * self._expert.mu_beta(x_test, i) for i in range(self.ydim)]
            if self._probit is not None:
                E_projected = [E_projected[i] + (prior_weight *
                                                 self._probit.project_point(self._expert.mu_beta(x_test, i))[:, 0])
                               for i in range(self.ydim)]

            for j in range(km):
                njm = np.sum(sIndexm[:, 0] == j)            # number of points in cluster j for state m
                if njm > 0:                                 # if not empty
                    logging.debug('feature cluster {0}'.format(j))
                    xjm = self.x[(sIndexm[:, 0] == j), :]
                    yjm = self.y[(sIndexm[:, 0] == j), :]
                    thetajm = thetam[j, :]

                    like_weight_j = (njm * np.exp(self._xlikelihood.log_predictive_marginal(x_test, xjm))) / \
                                    (am + self.n)
                    C += like_weight_j

                    mj, _ = self._expert.predict(xjm, yjm, x_test, thetajm)
                    if self._probit is None or return_latent is True:
                        E = [E[i] + (like_weight_j * mj[:, i]) for i in range(self.ydim)]
                    if self._probit is not None:  # project the marginal through the probit model
                        mj_projected = self._probit.project_point(mj)
                        E_projected = [E_projected[i] + (like_weight_j * mj_projected[:, i]) for i in range(self.ydim)]

        returnDict = dict()
        if 'E' in locals():
            returnDict['E'] = [E[i] / C for i in range(self.ydim)]
        if 'E_projected' in locals():
            returnDict['Ep'] = [E_projected[i] / C for i in range(self.ydim)]

        return returnDict

    def _predict_marginal_expectation(self, x_test, covariate, MC=None, burnin=0, importance_samples=15, thinning=1,
                                      return_latent=False):
        """
        Given a test point(s), find the expectation of the response across all states after burnin, conditional on only
        one covariate, with other covariates marginalised.

        :param x_test:      Input matrix for which we find the density of a given feature, averaged over all states.
            :type           2d-ndarray n_test * covariate dimension
        :param covariate:   The covariate index we are inputing. Other covariates are to be marginalised.
            :type           int
        :param burnin       number of Markov Chain Monte Carlo samples discarded for the burn-in period.
            :type           int

        :return:           the mean response of the JME model, marginalised over other covariates dimensions.
            :type
        """
        if MC is None:
            MC = self._mcmcSteps
        else:
            assert MC <= self._mcmcSteps, 'JmE._pred_marg_expec: MC={0} > completed {1}'.format(MC, self._mcmcSteps)

        assert len(self.states) > 1, 'JmE._pred_marg_expec: Need to update model at least once to sample GP hyper'
        assert MC > burnin, 'JmE._pred_marg_expec: #burnin > #MCMC samples. {0}!>{1}'.format(burnin, MC)
        assert x_test.ndim == 2, 'JmE._pred_marg_expec: Require 2d ndarray input x_test'
        assert x_test.shape[1] == 1, 'JmE._pred_marg_expec: Test covariate dimension must be 1'
        assert isinstance(covariate, (int,)), 'JmE._pred_marg_expec: Covariate must be an index integer'

        # Memory allocation
        C = np.zeros((x_test.shape[0]))                                          # Normalising constant
        if self._probit is None or return_latent is True:                        # Expectation
            E = [np.zeros(x_test.shape[0]) for _ in range(self.ydim)]
        if self._probit is not None:                                             # Expectation projected
            E_projected = [np.zeros(x_test.shape[0]) for _ in range(self.ydim)]

        # Pre-computations
        x_test_nan = np.empty((x_test.shape[0], self.xdim))
        x_test_nan[:] = np.nan
        x_test_nan[:, [covariate]] = x_test
        hxsp = np.exp(self._xlikelihood._log_full_marginal(x_test_nan[:, [covariate]]))      # Marginal likelihood

        for m in range(MC - burnin):
            pb.update_progress((m+1)/(MC - burnin), message='JmE._pred_marg_expec()')
            if np.mod(m, thinning):
                logger.debug('JmE._pred_marg_expec: Thinning by skipping m: {0}'.format(m + burnin))
                continue

            # Load state
            stateM = self.states[m + burnin]
            am = stateM['alpha']
            thetam = stateM['theta']
            sIndexm = stateM['sIndex']
            km = stateM['k_counter']

            # Prior contributions
            prior_weight = (am * hxsp) / (am + self.n)
            C += prior_weight
            if self._probit is None or return_latent is True:
                E = [E[i] + prior_weight * self._expert.mu_beta(x_test, i) for i in range(self.ydim)]
            if self._probit is not None:
                E_projected = [E_projected[i] +
                               (prior_weight * self._probit.project_point(self._expert.mu_beta(x_test, i))[:, 0])
                               for i in range(self.ydim)]

            # Posterior contributions
            for j in range(km):
                njm = np.sum(sIndexm[:, 0] == j)            # number of points in cluster j for state m
                if njm > 0:                                 # if not empty
                    logging.debug('feature cluster {0}'.format(j))
                    xjm = self.x[(sIndexm[:, 0] == j), :]
                    yjm = self.y[(sIndexm[:, 0] == j), :]
                    thetajm = thetam[j, :]

                    like_weight_j = (njm * np.exp(self._xlikelihood.log_predictive_marginal(x_test_nan, xjm, p=[covariate]))) / \
                                    (am + self.n)
                    C += like_weight_j

                    # Expectation of each cluster j (importance sampling, marg the local pred marg likelihoods)
                    if self._probit is None or return_latent is True:
                        integral = np.zeros((x_test.shape[0], 1))
                    if self._probit is not None:
                        integral_projected = np.zeros((x_test.shape[0], 1))
                    for sample in range(importance_samples):              # For each local predictive likelihood sample
                        # sample the local predictive likelihood
                        Xs = self._xlikelihood.sample_predictive_marginal(xjm, x_test.shape[0])
                        Xs[:, [covariate]] = x_test

                        mj, _ = self._expert.predict(xjm, yjm, Xs, thetajm)
                        if self._probit is None or return_latent is True:
                            integral += mj
                        if self._probit is not None:        # project the marginal expectation through the probit model
                            mj_projected = self._probit.project_point(mj)
                            integral_projected += mj_projected

                    if self._probit is None or return_latent is True:
                        integral /= importance_samples
                        E = [E[i] + (like_weight_j * integral[:, i]) for i in range(self.ydim)]
                    if self._probit is not None:
                        integral_projected /= importance_samples
                        E_projected = [E_projected[i] + (like_weight_j * integral_projected[:, i])
                                       for i in range(self.ydim)]

        returnDict = dict()
        if 'E' in locals():
            returnDict['E'] = [E[i] / C for i in range(self.ydim)]
        if 'E_projected' in locals():
            returnDict['Ep'] = [E_projected[i] / C for i in range(self.ydim)]

        return returnDict

    def _predict_density(self, x_test, y_test=None, MC=None, burnin=0, thinning=1, return_latent=False):
        """ Given a matrix of input vectors, find the density of a given feature, averaged over all states after the
        burn in period.

        :param x_test:                Input matrix for which we find the density of a given feature, for a given state.
            :type                     2d-ndarray n_test * input dim
        :param y_test:                The bins the density is calculated inside
        :param MC:                    The last state used of the MC chain
        :param burnin:                The first state used of the MC chain
        :param thinning:              The thinning factor (chain distance between MC states used in prediction)
        :param return_latent:         Return the latent density for when there is a probit model
        :return:                      The grid of x_test, y_test, and densities of y_test at x_test.
            :type                     Three 2d-ndarray:
                                            grids for covariate of interest,
                                            feature of interest values,
                                            density of those possible feature values.
        """
        if MC is None:
            MC = self._mcmcSteps
        else:
            assert MC <= self._mcmcSteps, 'JmE._pred_dens: MC={0} > completed {1}'.format(MC, self._mcmcSteps)

        assert len(self.states) > 1, 'JmE._pred_dens: Need to update model at least once to sample GP hyper'
        assert MC > burnin, 'JmE._pred_dens: burnin must be less than #MCMC samples. {0}!>{1}'.format(MC, burnin)
        assert x_test.ndim == 2, 'JmE._pred_dens: Require 2d ndarray input x_test'
        assert x_test.shape[1] == self.xdim, 'JmE._pred_dens: Test covariate dim must be equal to training dim'

        ##### Memory allocation
        #######################
        # Bins for y (limits fixed at extremes)
        if np.any(y_test) == None:
            rangeY_scaled = 0.2 * (np.max(self.y) - np.min(self.y))
            y_test = np.linspace(np.min(self.y) - rangeY_scaled, np.max(self.y) + rangeY_scaled, 1000)
        # Normalising constant
        C = np.zeros((x_test.shape[0]))
        # Density latent
        if self._probit is None or return_latent is True:
            D = [np.zeros((self.y.shape[1], x_test.shape[0])) for _ in range(self.ydim)]
        # Density projected
        if self._probit is not None:
            D_projected = [np.zeros((len(self._probit.categories), x_test.shape[0])) for _ in range(self.ydim)]

        ##### Pre-computations
        ######################
        # Marginal covariate likelihood
        hxs = np.exp(self._xlikelihood._log_full_marginal(x_test))
        # Marginal response likelihood
        if self._probit is None or return_latent is True:
            hyx_marg = self._expert.prior_marg_likelihood(x_test, y_test, samples=50)
        # Marginal response likelihood projected
        if self._probit is not None:
            hyx_marg_projected = self._expert.prior_marg_likelihood_probit(x_test, self._probit, samples=50)

        ##### Predict at each MC state
        ##############################
        for m in range(MC - burnin):
            pb.update_progress((m+1)/(MC - burnin), message='JmE._pred_dens()')

            # Thin MC chain
            if np.mod(m, thinning):
                logger.debug('JmE._pred_dens: Thinning by skipping m: {0}'.format(m + burnin))
                continue

            # Load state
            stateM = self.states[m + burnin]
            am = stateM['alpha']
            thetam = stateM['theta']
            sIndexm = stateM['sIndex']
            km = stateM['k_counter']

            # Prior contributions
            prior_weight = (am * hxs) / (am + self.n)
            C += prior_weight
            if self._probit is None or return_latent is True:
                D = [D[i] + (np.tile(prior_weight, (y_test.shape[0], 1)) * hyx_marg) for i in range(self.ydim)]
            if self._probit is not None:
                D_projected = [D_projected[i] + (np.tile(prior_weight, (len(self._probit.categories), 1)) *
                                                 hyx_marg_projected) for i in range(self.ydim)]

            # Posterior contributions
            for j in range(km):
                njm = np.sum(sIndexm[:, 0] == j)
                if njm > 0:  # if not empty
                    logging.debug('feature cluster {0}'.format(j))
                    xjm = self.x[(sIndexm[:, 0] == j), :]
                    yjm = self.y[(sIndexm[:, 0] == j), :]

                    thetajm = thetam[j, :]

                    like_weight_j = (njm * np.exp(self._xlikelihood.log_predictive_marginal(x_test, xjm))) / \
                                    (am + self.n)
                    C += like_weight_j

                    mj, sj = self._expert.predict(xjm, yjm, x_test, thetajm)
                    if self._probit is None or return_latent is True:
                        hyxL = np.transpose(scipy.stats.norm.pdf(y_test, mj, sj))
                        D = [D[i] + (np.tile(like_weight_j, (y_test.shape[0], 1)) * hyxL) for i in range(self.ydim)]
                    if self._probit is not None:
                        hyxL_projected = self._probit.project_density(mj.reshape((-1,)), sj.reshape((-1,)))
                        D_projected = [D_projected[i] +
                                       (np.tile(like_weight_j, (len(self._probit.categories), 1)) * hyxL_projected)
                                       for i in range(self.ydim)]

        returnDict = dict()
        returnDict['x_test'] = x_test
        returnDict['y_test'] = y_test
        if 'D' in locals():
            returnDict['D'] = [D[i] / np.tile(C, (y_test.shape[0], 1)) for i in range(self.ydim)]
        if 'D_projected' in locals():
            returnDict['Dp'] = [D_projected[i] / np.tile(C, (len(self._probit.categories), 1))
                                for i in range(self.ydim)]

        return returnDict

    def _predict_marginal_density(self, x_test, y_test=None, covariate=0, MC=None, burnin=0, importance_samples=15,
                                  thinning=1, return_latent=False):
        """
        Given a matrix of input vectors, find the density a given feature, averaged over all states after the burn in
        period. The density is found over a grid defined using the input dimension of x_test (using covariate) and
        linearly spaced y values (defined below).  x_test can be any set of inputs, but standard use is to set
        dimensions != covariate equal to the training mean.

        :param x_test:      Input matrix for which we find the density of a given feature, for a given state.
            :type           2d-ndarray n_test * 1
        :param covariate:   The index of the covariate we want to vary along. Needed for grids.
            :type           int
        :param feature:     The index of the feature whose density we want.
            :type           int

        :return:            the grid of x_test, y_test, and densities of y_test at x_test.
            :type           three 2d-ndarray, grids for covariate of interest, feature of interest values, and density
                            of those possible feature values.
        """
        if MC is None:
            MC = self._mcmcSteps
        else:
            assert MC <= self._mcmcSteps, 'JmE._pred_marg_dens: MC={0} > completed {1}'.format(MC, self._mcmcSteps)

        assert len(self.states) > 1, 'JmE._pred_marg_dens: Need to update model at least once to sample GP hyper'
        assert MC > burnin, 'JmE._pred_marg_dens: #burnin > #MCMC samples. {0}!>{1}'.format(burnin, MC)
        assert x_test.ndim == 2, 'JmE._pred_marg_dens: Require 2d ndarray input x_test'
        assert x_test.shape[1] == 1, 'JmE._pred_marg_dens: Test covariate dimension must be 1'
        assert isinstance(covariate, (int,)), 'JmE._pred_marg_dens: Covariate must be an index integer'

        # Memory allocation
        if np.any(y_test) == None:
            rangeY_scaled = 0.2 * (np.max(self.y) - np.min(self.y))         # Bins for y (limits fixed at extremes)
            y_test = np.linspace(np.min(self.y) - rangeY_scaled, np.max(self.y) + rangeY_scaled, 1000)
        Xgrid, Ygrid = np.meshgrid(x_test, y_test)                          # Grid for density (#y * #x)
        C = np.zeros((x_test.shape[0]))                                     # Normalising constant
        if self._probit is None or return_latent is True:                   # Density latent
            D = [np.zeros(Xgrid.shape) for _ in range(self.ydim)]
        if self._probit is not None:                                        # Density projected
            D_projected = [np.zeros((len(self._probit.categories), x_test.shape[0])) for _ in range(self.ydim)]

        # Pre-computations
        x_test_nan = np.empty((x_test.shape[0], self.xdim))
        x_test_nan[:] = np.nan
        x_test_nan[:, [covariate]] = x_test
        hxsp = np.exp(self._xlikelihood._log_full_marginal(x_test_nan[:, [covariate]]))   # Marginal covariate likelihood
        if self._probit is None or return_latent is True:                   # Approximate marginal likelihood (response)
            hyx_marg = self._expert.prior_marg_likelihood(x_test, y_test, samples=50)
        if self._probit is not None:
            hyx_marg_projected = self._expert.log_prior_marg_probit(x_test, self._probit, samples=50)

        for m in range(MC - burnin):
            pb.update_progress((m+1)/(MC - burnin), message='JmE._predict_marginal_density()')
            if np.mod(m, thinning):
                logger.debug('JmE._pred_marg_dens: Thinning by skipping m: {0}'.format(m + burnin))
                continue

            # Load state
            stateM = self.states[m + burnin]
            am = stateM['alpha']
            thetam = stateM['theta']
            sIndexm = stateM['sIndex']
            km = stateM['k_counter']

            # prior contributions
            prior_weight = (am * hxsp) / (am + self.n)
            C += prior_weight
            if self._probit is None or return_latent is True:
                D = [D[i] + (np.tile(prior_weight, (y_test.shape[0], 1)) * hyx_marg) for i in range(self.ydim)]
            if self._probit is not None:
                D_projected = [D_projected[i] + (np.tile(prior_weight, (len(self._probit.categories), 1)) *
                                                 hyx_marg_projected) for i in range(self.ydim)]

            # posterior contributions
            for j in range(km):
                njm = np.sum(sIndexm[:, 0] == j)
                if njm > 0:  # if not empty
                    logging.debug('feature cluster {0}'.format(j))
                    xjm = self.x[(sIndexm[:, 0] == j), :]
                    yjm = self.y[(sIndexm[:, 0] == j), :]

                    thetajm = thetam[j, :]

                    like_weight_j = (njm * np.exp(self._xlikelihood.log_predictive_marginal(x_test_nan, xjm,
                                                                                            p=[covariate]))) / \
                                    (am + self.n)
                    C += like_weight_j

                    # integral - approximate with importance sampling
                    # (local expectation of each cluster j, integrating out the local predictive marginal likelihoods)
                    if self._probit is None or return_latent is True:
                        integral = np.zeros((y_test.shape[0], x_test.shape[0]))
                    if self._probit is not None:
                        integral_projected = np.zeros((len(self._probit.categories), x_test.shape[0]))
                    for sample in range(importance_samples):  # for each local predictive likelihood sample
                        # sample the local predictive likelihood
                        Xs = self._xlikelihood.sample_predictive_marginal(xjm, x_test.shape[0])
                        Xs[:, [covariate]] = x_test

                        mj, sj = self._expert.predict(xjm, yjm, Xs, thetajm)
                        if self._probit is None or return_latent is True:
                            hyxL = np.transpose(scipy.stats.norm.pdf(y_test, mj, sj))
                            integral += hyxL
                        if self._probit is not None:
                            hyxL_projected = self._probit.project_density(mj.reshape((-1,)), sj.reshape((-1,)))
                            integral_projected += hyxL_projected

                    if self._probit is None or return_latent is True:
                        integral /= importance_samples
                        D = [D[i] + (np.tile(like_weight_j, (y_test.shape[0], 1)) * integral) for i in range(self.ydim)]
                    if self._probit is not None:
                        integral_projected /= importance_samples
                        D_projected = [D_projected[i] +
                                       (np.tile(like_weight_j, (len(self._probit.categories), 1)) * integral_projected)
                                       for i in range(self.ydim)]

        # test slice plots for different test inputs
        #import matplotlib.pyplot as plt
        #for xi in [6,8,10,12,14,16]:  # range(1):#len(x_test)-20):
        #    plt.plot(y_test, (D[0] / np.tile(C, (y_test.shape[0], 1)))[:, xi])
        #    plt.scatter(self._probit.categories,
        #                (D_projected[0] / np.tile(C, (len(self._probit.categories), 1)))[:, xi])
        #plt.grid()
        #plt.show(True)

        returnDict = dict()
        returnDict['Xgrid'] = Xgrid
        returnDict['Ygrid'] = Ygrid
        if 'D' in locals():
            returnDict['D'] = [D[i] / np.tile(C, (y_test.shape[0], 1)) for i in range(self.ydim)]
        if 'D_projected' in locals():
            returnDict['Dp'] = [D_projected[i] / np.tile(C, (len(self._probit.categories), 1))
                                for i in range(self.ydim)]

        return  returnDict
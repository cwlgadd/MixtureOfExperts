"""
A class for the Enriched Mixture of Experts model.

Author:
    Charles Gadd

Date:
    09/10/2017
"""
from __future__ import division         # for compatability between py2 and py3, better than casting division.
import itertools, inspect, sys, os
from MixtureOfExperts.utils import progress_bar as pb
from MixtureOfExperts.MCMC import split_merge as sm
import scipy.stats
from scipy.special import gammaln as lnGa
import numpy as np
import random
import copy
import pickle
import logging
logger = logging.getLogger(__name__)
import time

__all__ = ['EnrichedMixtureOfExperts']

class EnrichedMixtureOfExperts(object):
    """
    A class for the Enriched Mixture of Experts model.
    """

    # public (accessible through @property below)
    _X = None                          # NxD ndarray of covariates (missing data _not_ allowed)
    _Y = None                          # NxD ndarrray of features (missing data _not_ allowed)
    _alpha_theta = None                # concentration parameter for feature clusters
    _alpha_psi = None                  # concentration parameters for each covariate cluster.
    _MCMCstates = None                 # the chain of states from our algorithm. list of dictionaries

    # private
    _expert = None                     # The object used to model the experts. Must inherit certain functions
    _xlikelihood = None                # The likelihood on the inputs x
    _probit = None                     # optional probit model for ordinal/binary responses
    _latenty = None                    # latent y, used only when we have a probit model
    _mcmcSteps = None                  # number of Monte Carlo iterations that have been performed
    _mNeal8 = None                     # number of new clusters each reallocation step
    _k_counter = None                  # counter for number of feature clusters.
    _kl_counter = None                 # counter for number of covariate clusters in the jth feature.
    _sIndex = None                     # cluster allocation indexes for each data point.
    _theta = None                      # matrix containing hyper-parameters, updated rather than appended.
    _utheta = None
    _vtheta = None
    _upsi = None
    _vpsi = None
    _SM = None

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
        If we have a probit model, get the latent features, else get the raw features
        """
        if self._probit is None:                            # if we have no probit model work on raw features
            return self._Y
        elif self._latenty is not None:                     # if we have probit and latent available return latent
            return self._latenty
        else:                                               # if we have probit but latent not available return error
            raise ValueError('We have a probit model, but have not calculated the latent response yet.')

    @property
    def ydim(self):
        """
        Get the number of feature dimensions
        """
        return self._Y.shape[1]

    @property
    def alpha_theta(self):
        """
        Get the concentration parameter alpha_theta
        """
        return self._alpha_theta

    @property
    def k_unique(self):
        """
        Get the number of unique feature clusters.
        """
        return len(np.unique(self._sIndex[:, 0]))

    @property
    def kl_unique(self):
        """
        Get the number of unique covariate clusters for each feature cluster.
        """
        kl = np.zeros((self.n+1))
        featureClusters = self._k_counter
        for clusterj in range(featureClusters):
            indexJ = np.where(self._sIndex[:, 0] == clusterj)
            kl[clusterj] = len(np.unique(self._sIndex[indexJ, 1]))

        return kl

    @property
    def kx1plus(self):
        """
        return:                 the number of x-clusters with more than one data point
                                their indexes
        """
        kx1plus = 0
        indexes = []
        for i in np.unique(self._sIndex[:, 0]):                                      # each each y cluster
            for j in np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1]):          # each sub x-cluster in each y
                xiny = np.sum((self._sIndex[:, 0] == i) & (self._sIndex[:, 1] == j)) # count how many in sub-cluster
                if xiny > 1:                                                         # if more than one data point
                    kx1plus += 1                                                     # add 1 for the x-cluster
                    indexes.append([i, j])                                           # and record the index

        # x2p = [len(np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1]))
        #       if len(np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1])) >= 2 else 0
        #       for i in range(self._k_counter)]
        #
        # return np.sum(x2p)

        return kx1plus, indexes

    @property
    def kx2plus(self):
        """
        return:                 the number of x-clusters within a y-cluster with more than one x-cluster
                                their indexes
        """
        kx2plus = 0
        indexes = []
        for i in np.unique(self._sIndex[:, 0]):                                  # for each y cluster
            if len(np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1])) > 1:   # if there are 2+ unique x clusters
                for j in np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1]):  # loop over them
                    kx2plus += 1                                                 # add 1 for every unique
                    indexes.append([i, j])                                       # and record the index

        return kx2plus, indexes

    def subcluster(self, yindex, xindex):
        """
        Given the y and x|y cluster indices, return the sub-cluster's covariates and indices of samples in that cluster
        """
        xlj = self.x[(self._sIndex[:, 0] == yindex) & (self._sIndex[:, 1] == xindex), :]
        xlj_nidx = np.where((self._sIndex == [yindex, xindex]).all(axis=1))[0]

        return xlj, xlj_nidx

    @property
    def state(self):
        """
        Get the current state of our class
        """
        if self._mcmcSteps == 0:            # Need to update to get latent samples
            latent = self._Y
        else:
            latent = self.y

        current_state = {'k_counter': self._k_counter,
                         'kl_counter': self._kl_counter,
                         'sIndex': self._sIndex,
                         'theta': self._theta,
                         'alpha_theta': self._alpha_theta,
                         'alpha_psi': self._alpha_psi,
                         'latent': latent}

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
            indexMatrix_covariate = np.zeros((self._mcmcSteps, self.n))

            for sample in range(self._mcmcSteps):
                state = self.states[sample]['sIndex']
                indexMatrix_feature[sample, :] = state[:, 0]
                indexMatrix_covariate[sample, :] = state[:, 1]
        else:
            indexMatrix_feature = self._sIndex[:, 0]
            indexMatrix_covariate = self._sIndex[:, 1]

        return indexMatrix_feature, indexMatrix_covariate

    def __init__(self, x, y, expert, xlikelihood, probit=None, mneal8=4, init_f=6, init_c=6, SM=True, name='Enriched'):
        """
        Initialize the model.

        :param x:               Observed features
            :type                np.ndarray (#samples * #features)
        :param y:               Observed covariates
            :type                np.ndarray (#samples * #covariates)
        """

        self.__name__ = name
        self._SM = SM

        try:
            self.load()
            logger.info('Loaded Enriched Mixture of Experts model from root directory {0}, continuing'.format(os.getcwd()))
        except:
            logger.debug('Creating new Enriched Mixture of Experts model')
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

            # Monte Carlo parameterisations and intialisations
            self._mcmcSteps = 0
            self._MCMCstates = []
            self._mNeal8 = mneal8

            # Memory allocation, we have at most n clusters, +1 is for overflow new if in extreme case of n singletons.
            self._theta = np.zeros((self.n + 1, self._expert.n_hyper))         # GP hyper-parameter
            self._alpha_psi = np.zeros(self.n + 1)                             # Covariate concentration parameter
            self._kl_counter = np.zeros(self.n + 1)                            # Covariate counters
            self._sIndex = np.zeros((self.n, 2))                               # Allocation index, col1=feature, col2=input

            """ prior for concentration parameter alpha"""
            self._utheta = 1
            self._vtheta = 1
            self._upsi = 1
            self._vpsi = 1

            """ initialise feature clusters """
            if init_f == 'singleton':                                        # all points initially allocated to own cluster
                self._sIndex[:, 0] = np.arange(self.n)
            elif isinstance(init_f, int):                                    # sample uniformly from np.arange(init_f)
                self._sIndex[:, 0] = np.random.choice(init_f, self.n)
            elif len(init_f) == self.n:                                      # allocate to specified feature clusters
                self._sIndex[:, 0] = init_f
            else:
                logging.critical('Exiting. Unable to intialise feature allocations with init_f {0}'.format(init_f))
                sys.exit()
            self._k_counter = int(np.max(np.unique(self._sIndex[:, 0]))+1)   # number of unique feature clusters

            """ initialise covariate clusters """
            for clusterj in range(self._k_counter):
                indexJ = np.where(self._sIndex[:, 0] == clusterj)[0]         # ind of points in this feature cluster
                if len(indexJ) > 0:
                    if init_c == 'singleton':                                # all points initially allocated to own cluster
                        self._sIndex[indexJ, 1] = np.arange(len(indexJ))
                    elif isinstance(init_c, int):                            # sample uniformly from np.arange(init_c)
                        self._sIndex[indexJ, 1] = np.random.choice(init_c, len(indexJ))
                    self._kl_counter[clusterj] = int(np.max(np.unique(self._sIndex[indexJ, 1])))+1 # number unique covariate

            """ set parameters describing intitial state by sampling posteriors & priors """
            rows = np.asarray(np.unique(self._sIndex[:, 0]), dtype=np.int16)  # which feature clusters are used for sampling
            self._alpha_theta = np.random.gamma(self._utheta, 1./self._vtheta)                     # feature clusters
            self._alpha_psi[rows] = np.random.gamma(self._upsi, 1./self._vpsi, size=len(rows))   # covariate clusters
            self._theta[rows, :] = self._expert.prior(samples=len(np.unique(self._sIndex[:, 0])))

            """ if we are using a probit model and unfixed nugget init high GP variance for truncated sampling """
            if self._probit is not None and len(self._expert.index_nugget) > 0:
                # assert self._expert._constrain_nugget is False
                for feature in range(self.ydim):
                    itershift = feature * self._expert._n_hyper_indGP
                    self._theta[rows, itershift + self._expert.index_nugget] = 1*np.std(self._Y[:, feature])

            """ add this first state to chain """
            self._append_state()                                              # initial state at 0 index

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = "\nMODEL: " + self.__name__ + '\n'
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

        for _ in itertools.repeat(None, samples):
            logger.info('Iteration {0}'.format(self._mcmcSteps+1))

            # non-continuous output through data augmentation, sample latent y
            if self._probit is not None:
                logger.info('Sampling probit model')
                self._latenty = self._probit.sample_latent_response(self._X, self._sIndex[:, 0], self._Y, self._theta,
                                                                    position=self._latenty)
            self._remove_empty_()
            self.check_consistent('test: remove empty each iteration, instead of waiting to fill')

            # Local moves, gibbs updates
            self._update_allocations()

            # Global moves
            self._Metropolis_steps()
            if self.xdim >= 0 and self._SM is not False:
                SM = sm.SplitMerge(self)                            # Create split-merge, pass self by reference
                SM()                                                # Perform SM moves, edits this as by references
                self.check_consistent('SM()')

            # Hyper updates
            self._update_hyperparameters()
            self._update_concentration()

            # save new step
            self._append_state()                                                # save new step
            self._mcmcSteps += 1
            self.check_consistent('call()')

    def save(self, root=''):
        """
        Save class as 'self.__name__'.xml. Includes classes defined in the hierarchy, such as expert.
        """
        with open(root + self.__name__+'.xml', 'wb') as selfFile:
            logger.info('Saving to {0}'.format(root + self.__name__ + '.xml'))
            pickle.dump(self.__dict__, selfFile, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug('Saved model')

    def load(self, root=''):
        """
        Load class from 'self.__name__'.xml
        """
        with open(root + self.__name__ +'.xml', 'rb') as selfFile:
            logger.info('Loading from {0}'.format(root + self.__name__ + '.xml'))
            self.__dict__.update(pickle.load(selfFile))
            logger.debug('Loaded model')

    def _update_hyperparameters(self):
        """
        update the cluster hyper parameters by sampling the posterior using Hamiltonian Monte Carlo
        """
        logger.info("updating hyper parameters for {0} cluster(s), nK={1}".format(len(np.unique(self._sIndex[:, 0])),
                                                                                  self._k_counter))

        for k in range(self._k_counter):                         # for each feature cluster.
            if sum(self._sIndex[:, 0] == k) == 0:                # if empty don't update and remove from previous.
                self._theta[k, :] = np.zeros((self._theta.shape[1]))
            else:                                                # else update, initialising at last value
                theta_old = copy.copy(self._theta[k, :])

                if self._mcmcSteps < 10:                         # For first few iterations initialise with last set of hyper-parameters
                    init = False #self._theta[k, :]
                else:                                            # When random initialisation are gone we can use MML
                    init = True

                self._theta[k, :], chain = self._expert.posterior(self.x[self._sIndex[:, 0] == k, :],
                                                                  self.y[self._sIndex[:, 0] == k, :],
                                                                  init,
                                                                  samples=50,
                                                                  stepsize=0.05)  # stepsize=0.05 , stepsize_range=[0.05, 20], #[5e-2, 1e0]#

                # debugging for priors
                #self._theta[k, self._expert._index_lengthscale[0]] = 1
                #self._theta[k, self._expert._index_lengthscale[1:]] = 30

                # plot posterior (debugging)
                #import matplotlib.pyplot as plt
                #for iter in range(chain[0].shape[1]):
                #    plt.subplot(211)
                #    plt.plot(np.linspace(1, chain[0].shape[0] - 1, chain[0].shape[0]), chain[0][:, iter])
                #    plt.subplot(212)
                #    plt.hist(chain[0][:, iter])
                #    plt.show()

                assert np.all(self._theta[k, :] != theta_old), \
                    'MC did not mix: cluster {0}, samples:\n {1} \n{2}'.format(k, theta_old, self._theta[k, :])
                logging.debug('hyper-parameters for cluster {0}: {1}'.format(k, self._theta[k, :]))

        self.check_consistent('update hyper')

    def _update_allocations(self):
        """
        Update allocation variables via a Polya Urn process.

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

        for i in order:                                                         # Update allocations for each data point
            logger.debug("\t sample {0} with latent response {1} target {2}".format(i, self.y[i, :], self._Y[i, :]))

            xi = np.reshape(self.x[i, :], (1, -1))
            yi = np.reshape(self.y[i, :], (1, -1))
            currentIndex = self._sIndex[i, 0]                                        # To check if cluster empty at end
            # memory allocation
            logap = -np.inf * np.ones((self._k_counter + self._mNeal8, self.n+1))    # log alloc prob, memory alloc
            gploglikxi = np.zeros(self._k_counter)                                   # GP log input_models, cached

            """ allocation probability for each occupied cluster (excluding sample i) """
            for j in range(int(self._k_counter)):                               # Existing feature clusters

                sjIndex = (self._sIndex[:, 0] == j)   # boolean for samples in jth feature cluster
                sjmiIndex = np.delete(sjIndex, i, 0)  # as above, minus current sample i
                nkmi = sum(sjmiIndex)                 # #samples in f cluster j, excluding i.

                if nkmi == 0:
                    pass                    # if singleton or empty cluster, don't allocate back into its own cluster.
                                            # Instead, we use this cluster as one of the mNeal8 proposed clusters.
                else:
                    # get feature cluster datum, minus xi -> Xk-i and Yk-i
                    Xkmi = np.delete(self.x, i, 0)[sjmiIndex]
                    Ykmi = np.delete(self.y, i, 0)[sjmiIndex]

                    # Expert likelihood - GP predictive likelihood of data vector i
                    gploglikxi[j] = self._expert.log_pred_marg_likelihood(Xkmi, Ykmi, xi, yi, self._theta[j, :])

                    for l in range(int(self._kl_counter[j])):                   # covariate clusters
                        # points in f cluster's c cluster, excluding point i (if i is in cluster j).

                        slIndex = (self._sIndex[:, 1] == l)  # boolean for samples in lth covariate cluster
                        slmiIndex = np.delete(slIndex, i, 0)  # as above, minus current sample i

                        # number of samples (-i) in subcluster.
                        #       Logical 'and' does need to be computed later. As it is expensive, use this quicker
                        #       indexing approach here and compute it only when nljmi is non-zero. TODO: is it necessary later?
                        nljmi = sum(np.delete(self._sIndex, i, 0)[sjmiIndex, 1] == l)
                        #nljmi_test = sum(np.logical_and(sjmiIndex, slmiIndex))

                        if nljmi == 0:          # if singleton covariate cluster, don't allocate back into itself.
                            pass
                        else:                   # get allocation probability
                            # X marginal likelihood
                            Xljmi  = np.delete(self.x, i, 0)[np.logical_and(sjmiIndex, slmiIndex), :]

                            # Allocation probabilities
                            logap[j, l] = np.log(nkmi * nljmi) - \
                                          np.log(self._alpha_psi[j] + nkmi) + \
                                          gploglikxi[j] + \
                                          self._xlikelihood.log_predictive_marginal(xi, Xljmi)

                            if not np.isfinite(logap[j, l]):
                                logging.warning('acceptance probability is not finite: {0}'.format(logap[j, l]))

                    logap[j, int(self._kl_counter[j])] = np.log(nkmi * self._alpha_psi[j]) - \
                                                         np.log(self._alpha_psi[j] + nkmi) + \
                                                         gploglikxi[j] + \
                                                         self._xlikelihood.log_marginal(xi)
                    if not np.isfinite(logap[j, int(self._kl_counter[j])]):
                        logging.warning('acceptance probability is not finite: {0}'.format(logap[j, int(self._kl_counter[j])]))


            """ Allocation probability for mNeal8 new (and current if singleton) y-clusters by sampling priors """
            # sample new clusters
            new_theta = self._expert.prior(self._mNeal8)                         # samples * n_hyper
            if np.sum(self._sIndex[:, 0] == self._sIndex[i, 0]) == 1:            # replace first with singleton
                new_theta[0, :] = self._theta[int(self._sIndex[i, 0]), :]
            # allocation probability for new cluster (and existing cluster if a singleton).
            for im in range(self._mNeal8):                                       # allocation probability
                logap[int(self._k_counter) + im, 0] = np.log(float(self._alpha_theta)) - \
                                                      np.log(self._mNeal8) + \
                                                      self._expert.log_marg_likelihood(new_theta[im, :], xi, yi) + \
                                                      self._xlikelihood.log_marginal(xi)

                assert np.isfinite(logap[int(self._k_counter) + im, 0]), \
                    'log acceptance probability is not finite: {0}'.format(logap[int(self._k_counter) + im, 0])

            """ exp-normalize trick """
            logap -= np.max(logap[logap != -np.inf])

            """ combine all allocation probabilities and normalise """
            apnorm = np.exp(logap) / sum(sum(np.exp(logap)))
            logging.debug('\t apnorm above 0.1: {0}'.format(apnorm[apnorm > 0.1]))

            try:
                sumapnorm = sum(sum(apnorm))
                assert np.isclose(sumapnorm, 1), "alloc prob does not sum to one: {0}".format(sumapnorm)
            except:
                logging.critical(logap)
                logging.critical(apnorm)
                logging.critical([type(iter) for iter in apnorm])
                logging.critical('Unable to sum over normalised probabilities. Ensure they are finite numbers. \n {0} \n {1}'.format(logap, apnorm))
                raise SystemExit(1)

            apcum = np.cumsum(np.ndarray.flatten(apnorm))

            """ Allocate point and update cluster parameters """
            apbool = scipy.stats.uniform(0, 1).rvs(1) > apcum
            idx = np.min(np.where(apbool == 0))
            idx_alloc_X = int(idx % logap.shape[1])
            idx_alloc_Y = int((idx - idx_alloc_X) / logap.shape[1])

            if idx_alloc_Y < self._k_counter:                                    # allocate xi to an existing cluster
                logger.debug("\t ... to an existing feature cluster, index #{0}".format(idx_alloc_Y))

                self._sIndex[i, :] = [idx_alloc_Y, idx_alloc_X]
                # theta and self._k_counter do not change

                # print results and edit counters if required
                if idx_alloc_X < self._kl_counter[idx_alloc_Y]:
                    logger.debug("\t\t ... to an existing covariate cluster, index #{0}".format(idx_alloc_X))

                    # self._kl_counter does not change
                else:
                    logger.debug("\t\t ... to an new covariate cluster, index #{0}".format(idx_alloc_X))

                    self._kl_counter[idx_alloc_Y] = self._kl_counter[idx_alloc_Y] + 1          # update self._kl_counter

                #logger.debug("allocated to ({0},{1})".format(idx_alloc_Y, idx_alloc_X))
                #logger.debug("with new counters ({0},{1})".format(self._k_counter, self._kl_counter[idx_alloc_Y]))
            else:                                                               # allocate xi to a new cluster
                logger.debug("\t ... to mNeal8 cluster, proposed index #{0}".format(idx_alloc_Y - self._k_counter))
                if np.sum(self._sIndex[:,0] == self._sIndex[i,0]) == 1 and idx_alloc_Y == self._k_counter:
                        logger.debug("\t\t ... which is the same singleton cluster")

                # add new cluster and update allocation for xi.
                self._sIndex[i, :] = [self._k_counter, 0]
                self._alpha_psi[self._k_counter] = scipy.stats.gamma.rvs(self._upsi, self._vpsi)
                self._theta[self._k_counter, : ] = new_theta[idx_alloc_Y - self._k_counter, :]
                self._kl_counter[self._k_counter] = 1
                self._k_counter = self._k_counter + 1

                #logger.debug("allocated to ({0},{1})".format(self._k_counter-1, 0))
                #logger.debug("with new counters ({0},{1})".format(self._k_counter, self._kl_counter[self._k_counter-1]))

            """ care taking - remove parameterisation for emptied clusters """
            if sum(self._sIndex[:,0] == currentIndex) == 0:                       # original feature cluster emptied
                self._theta[int(currentIndex), :] = np.zeros((self._theta.shape[1]))
                self._alpha_psi[int(currentIndex)] = 0

            """ care taking - if full remove empty - emptied cluster index's are removed and k counters adjusted """
            if self._k_counter > self.n or (self._kl_counter >= self.n).any():
                logger.info("\t\t\t ... removing empty clusters")
                self._remove_empty_()

        # assertions
        self.check_consistent('update allocations')

        logger.debug(f'Update allocations took {time.time() - start_time} seconds')

    def _Metropolis_steps(self):
        """
        Propose moving an x-cluster to be nested within a different or new y-cluster. This step is split into three
        possible moves:
            (1) an x-cluster, among those within y-clusters with more than one x-cluster, is moved to a different
                y-cluster
            (2) an x-cluster, among those within y-clusters with more than one x-cluster, is moved to a new y-cluster.
            (3) an x-cluster, among those within y-clusters with only one x-cluster, is moved to a different y-cluster.

        This is a wrapper for the steps split into separate functions below. This is to make the code more readable.
        When calculating Gamma functions we can use a log transformation without scipy.special.gammasgn as the arguments
        are >=0 => that Ga(arguments) >= 0 and so scipy.special.gammasgn == 1
        """
        logger.info('Performing Metropolis Hastings mixing steps')

        self._Metropolis_one()

        # If we have empty sets then we make the moves with probability one
        x1 = [1 if len(np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1])) == 1
              else 0 for i in range(self._k_counter)]
        kx1 = np.sum(x1)
        if kx1 == 0:
            self._Metropolis_two()
            return
        x2p = [len(np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1]))
               if len(np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1])) >= 2
               else 0 for i in range(self._k_counter)]
        kx2p = np.sum(x2p)
        if kx2p == 0:
            self._Metropolis_three()
            return

        # Else we have the normal case where moves are chosen with equal probability.
        if scipy.stats.uniform(0, 1).rvs(1) < 0.5:
            self._Metropolis_two()
        else:
            self._Metropolis_three()

    def _Metropolis_one(self):
        """
        Propose moving an x-cluster to be nested within a different. Step one:
            (1) an x-cluster, among those within y-clusters with more than one x-cluster, is moved to a different
                y-cluster.
        """

        """ Select a feature cluster to move from, equal weight """
        # Get the number of x clusters within a y cluster with more than one x cluster
        # loop over feature clusters, then #covariates if more than one unique, else 0
        x2p = [len(np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1]))
               if len(np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1])) >= 2
               else 0 for i in range(self._k_counter)]
        kx2p = np.sum(x2p)
        if kx2p == 0:
            logger.info('\t ... skipping Metropolis step one, no y clusters with more than one x cluster.')
            return
        # first choose which feature this covariate cluster belongs from, weighted by number in that cluster.
        cumsum = np.cumsum( x2p / kx2p )
        cumbool =  scipy.stats.uniform(0, 1).rvs(1) > cumsum
        j = np.min(np.where(cumbool == 0))
        # number of points in feature cluster j
        Nj = np.shape(self._sIndex[(self._sIndex[:, 0] == j), :])[0] # np.sum(self._sIndex[:, 0] == j)
        assert Nj > 0, str(Nj)
        # points in feature cluster j
        Xj = self.x[self._sIndex[:, 0] == j, :]
        Yj = self.y[self._sIndex[:, 0] == j, :]
        assert Xj.shape[0] == Yj.shape[0] == Nj
        #print('j: {0}, Nj: {1}'.format(j, Nj))

        """ Select an x cluster from this, equal weight """
        # then uniformly choose which covariate cluster inside that cluster will be randomly moved.
        l = np.random.choice(np.unique(self._sIndex[self._sIndex[:, 0] == j, 1]))
        # Choosing sub-cluster l in feature cluster j to propose moving
        Xlj = self.x[(self._sIndex[:, 0] == j) & (self._sIndex[:, 1] == l), :]
        Ylj = self.y[(self._sIndex[:, 0] == j) & (self._sIndex[:, 1] == l), :]
        # With this many samples
        Nlj = np.shape(Xlj)[0]
        assert Nlj > 0, str(Nlj)
        #print('l: {0}, Nlj: {1}'.format(l, Nlj))

        """ Select which y cluster this can be moved to. This is chosen with probability proportional to the
            conditional marginal likelihood. Do not allocate to same, if this is the only option, then skip... """
        if self.k_unique == 1:
            logger.info('\t ... skipping Metropolis step one, no second y cluster to move to.')
            return
        # loop over unique feature co-ordinates in sIndex, keeping those not j.
        H = [i for i in np.unique(self._sIndex[:, 0]) if i != j]        # Set of feature clusters to considered moving to
        probH = np.asarray([self._expert.log_pred_marg_likelihood(self.x[self._sIndex[:, 0] == h, :],
                                                                  self.y[self._sIndex[:, 0] == h, :],
                                                                  Xlj, Ylj, self._theta[int(h), :]) for h in H])
        probH -= np.max(probH[probH != -np.inf])                        # exp-normalize trick
        probH = np.exp(probH) / sum(np.exp(probH))                      # combine all probabilities and normalise
        probH = np.cumsum(probH)                                        # cumulative sum of probabilities
        apbool = scipy.stats.uniform(0, 1).rvs(1) > probH               # Choose point h in H with probH
        idx = np.min(np.where(apbool == 0))
        h = int(H[int(idx)])                                            # and continue with chosen point
        Nh = np.shape(self._sIndex[(self._sIndex[:, 0] == h), :])[0]    # number of points in feature cluster h
        assert Nh > 0, str(Nh)
        Xh = self.x[self._sIndex[:, 0] == h, :]                         # points in feature cluster h
        Yh = self.y[self._sIndex[:, 0] == h, :]
        assert Xh.shape[0] == Yh.shape[0] == Nh
        #print('h: {0}, Nh: {1}'.format(h, Nh))

        """ Proposal state """
        # data in cluster j without points from  covariate cluster l
        Xjs = self.x[(self._sIndex[:, 0] == j) & (self._sIndex[:, 1] != l), :]
        Yjs = self.y[(self._sIndex[:, 0] == j) & (self._sIndex[:, 1] != l), :]
        assert Xjs.shape[0] == Yjs.shape[0] == Nj - Nlj
        # data in cluster h and from covariate cluster l inside of j
        Xhs = self.x[(self._sIndex[:, 0] == h) | (self._sIndex[:, 0] == j) & (self._sIndex[:, 1] == l), :]
        Yhs = self.y[(self._sIndex[:, 0] == h) | (self._sIndex[:, 0] == j) & (self._sIndex[:, 1] == l), :]
        assert Xhs.shape[0] == Yhs.shape[0] == Nh + Nlj
        # number of x clusters within a y cluster with more than one x cluster under proposal
        sIndex_prop = copy.deepcopy(self._sIndex)
        # move points from lj to h
        sIndex_prop[(sIndex_prop[:, 0] == j) & (sIndex_prop[:, 1] == l), :] = [h, self._kl_counter[h]]
        # Get the number of x clusters within a y cluster with more than one x cluster under proposal
        # loop over feature clusters, then #covariates if more than one unique, else 0
        x2ps = [len(np.unique(sIndex_prop[(sIndex_prop[:, 0] == i), 1]))
               if len(np.unique(sIndex_prop[(sIndex_prop[:, 0] == i), 1])) >= 2
               else 0 for i in range(self._k_counter)]
        kx2ps = np.sum(x2ps)

        """ Allocation probability """
        # fractions, split for notational convenience.
        LogGa_frac1 = lnGa(Nj - Nlj) + lnGa(Nh + Nlj) - lnGa(Nj) - lnGa(Nh)
        LogGa_frac2 = lnGa(self._alpha_psi[j] + Nj) + lnGa(self._alpha_psi[h] + Nh) - \
                      lnGa(self._alpha_psi[j] + Nj - Nlj) - lnGa(self._alpha_psi[h] + Nh + Nlj)
        psi_frac = self._alpha_psi[h] / self._alpha_psi[j]
        kfrac = kx2p / kx2ps
        # TODO: cost can be reduced by re-using probabilities from h allocation for a small improvement in speed.
        likelihood_num = np.sum([np.exp(self._expert.log_pred_marg_likelihood(self.x[self._sIndex[:, 0] == htilde, :],
                                                                           self.y[self._sIndex[:, 0] == htilde, :],
                                                                           Xlj, Ylj, self._theta[int(htilde), :]))
                                     for htilde in np.unique(self._sIndex[:, 0]) if htilde != j])
        likelihood_denom = np.sum([np.exp(self._expert.log_pred_marg_likelihood(self.x[sIndex_prop[:, 0] == htilde, :],
                                                                             self.y[sIndex_prop[:, 0] == htilde, :],
                                                                             Xlj, Ylj, self._theta[int(htilde), :]))
                                     for htilde in np.unique(self._sIndex[:, 0]) if htilde != h])
        likelihood = likelihood_num / likelihood_denom
        #print(likelihood_num)
        #print(likelihood_denom)

        acceptance = np.exp(LogGa_frac1) * np.exp(LogGa_frac2) * psi_frac * likelihood * kfrac
        assert acceptance >= 0, \
            "negative acceptance probability: {0}, ({1},{2},{3},{4},{5},{6})".format(acceptance, LogGa_frac1,
                                                                                     LogGa_frac2, psi_frac,
                                                                                     likelihood_num,
                                                                                     likelihood_denom, kfrac)

        if scipy.stats.uniform(0, 1).rvs(1) < acceptance:
            logger.info('\t ... accepting Metropolis step one with probability {0}'.format(acceptance))
            self._sIndex = sIndex_prop
            self._kl_counter[h] += 1

            self.check_consistent('M1')
        else:
            logger.info('\t ... rejecting Metropolis step one with probability {0}'.format(acceptance))

    def _Metropolis_two(self):
        """
        Step two:
            (2) an x-cluster, among those within y-clusters with more than one x-cluster, is moved to a new y-cluster.
        """

        # print(self._sIndex)

        """ Select a feature cluster to move from, biased proportional to number of covariates """
        # Get the number of x clusters within a y cluster with more than one x cluster
        # loop over feature clusters, then #covariates if more than one unique, else 0
        x2p = [len(np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1]))
               if len(np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1])) >= 2
               else 0 for i in range(self._k_counter)]
        kx2p = np.sum(x2p)
        if kx2p == 0:
            logger.info('\t ... skipping Metropolis step two, no y clusters with more than one x cluster.')
            return
        # choose which feature this covariate cluster belongs from, weighted by number in that cluster.
        cumsum = np.cumsum(x2p / kx2p)
        cumbool = scipy.stats.uniform(0, 1).rvs(1) > cumsum
        j = np.min(np.where(cumbool == 0))
        Nj = np.shape(self._sIndex[(self._sIndex[:, 0] == j), :])[0]            # number of points in feature cluster j
        assert Nj > 0, str(Nj)
        Xj = self.x[self._sIndex[:, 0] == j, :]                                 # points in feature cluster j
        Yj = self.y[self._sIndex[:, 0] == j, :]
        assert Xj.shape[0] == Yj.shape[0] == Nj
        #print('j: {0}, Nj: {1}'.format(j, Nj))

        """ Select an x cluster from this, equal weight """
        # then uniformly choose which covariate cluster inside that cluster will be randomly moved.
        l = np.random.choice(np.unique(self._sIndex[self._sIndex[:, 0] == j, 1]))
        # Choosing sub-cluster l in feature cluster j to propose moving
        Xlj = self.x[(self._sIndex[:, 0] == j) & (self._sIndex[:, 1] == l), :]
        Ylj = self.y[(self._sIndex[:, 0] == j) & (self._sIndex[:, 1] == l), :]
        Nlj = np.shape(Xlj)[0]                                                  # With this many saples
        assert Nlj > 0, str(Nlj)
        # print('l: {0}, Nlj: {1}'.format(l, Nlj))

        """ Propose new y cluster """
        # sample new cluster parameters from the priors
        theta_prop = self._expert.prior(1)                                      # n_hyper
        alpha_psi = scipy.stats.gamma.rvs(self._upsi, self._vpsi, size=1)

        """ Proposal state """
        Xkp1s = Xlj                 # new proposed cluster is the cluster l inside of j
        Ykp1s = Ylj
        sIndex_prop = copy.deepcopy(self._sIndex)         # proposal indexing, moving points from l|j to new cluster
        sIndex_prop[(sIndex_prop[:, 0] == j) & (sIndex_prop[:, 1] == l), :] = [self._k_counter, 0]
        # Get the number of x clusters within a y cluster with exactly one x cluster under proposal
            # (number of y clusters with one x cluster)
        # loop over feature clusters, then #covariates if one unique, else 0
        x1s = [len(np.unique(sIndex_prop[(sIndex_prop[:, 0] == i), 1]))
               if len(np.unique(sIndex_prop[(sIndex_prop[:, 0] == i), 1])) == 1
               else 0 for i in range(self._k_counter+1)]
        kx1s = np.sum(x1s)

        # fractions, split for notational convenience.
        # Ga(1) = 1 and Ga(x+1) = xGa(x) (cancellation) properties saves computation on new cluster
        LogGa_frac1 = lnGa(Nj - Nlj) + lnGa(Nlj) - lnGa(Nj)
        LogGa_frac2 = lnGa(self._alpha_psi[j] + Nj) + lnGa(alpha_psi) - \
                      lnGa(self._alpha_psi[j] + Nj - Nlj) - lnGa(alpha_psi + Nlj)
        psi_frac = self._alpha_theta * (alpha_psi / self._alpha_psi[j])
        # TODO: Check again
        log_likelihood_num = self._expert.log_marg_likelihood(theta_prop[0, :], X=Xkp1s, Y=Ykp1s)
        likelihood_denom =  np.sum([np.exp(self._expert.log_pred_marg_likelihood(self.x[self._sIndex[:, 0] == h, :],
                                                                                 self.y[self._sIndex[:, 0] == h, :],
                                                                                 Xlj, Ylj, self._theta[int(h), :]))
                                        for h in np.unique(self._sIndex[:, 0])])
        #print(log_likelihood_num)
        #print(f'denominator likelihood: {likelihood_denom}, multi-variate of dimension {Nlj}')
        likelihood = np.exp(log_likelihood_num) / likelihood_denom
        kfrac = kx2p / (kx1s * self.k_unique)


        acceptance = np.exp(LogGa_frac1) * np.exp(LogGa_frac2) * psi_frac * likelihood * kfrac
        assert acceptance >= 0, \
            "negative acceptance probability: {0}, ({1},{2},{3},{4},{5})".format(acceptance, LogGa_frac1, LogGa_frac2,
                                                                                 psi_frac, likelihood, kfrac)

        if scipy.stats.uniform(0, 1).rvs(1) < acceptance:
            logger.info('\t ... accepting Metropolis step two with probability {0}'.format(acceptance))
            self._sIndex = sIndex_prop
            self._kl_counter[self._k_counter] += 1
            self._theta[self._k_counter, :] = theta_prop
            self._alpha_psi[self._k_counter] = alpha_psi
            self._k_counter += 1

            """ care taking - if full remove empty - emptied cluster index's are removed and k counters adjusted """
            if self._k_counter > self.n or (self._kl_counter >= self.n).any():
                logger.info("\t\t\t ... removing empty clusters")
                self._remove_empty_()

            assert self._kl_counter[self._k_counter - 1] == 1
            self.check_consistent('M2')
        else:
            logger.info('\t ... rejecting Metropolis step two with probability {0}'.format(acceptance))

    def _Metropolis_three(self):
        """
        Propose moving an x-cluster to be nested within a different or new y-cluster. Step three:
            (3) an x-cluster, among those within y-clusters with only one x-cluster, is moved to a different y-cluster.
        """

        #print(self._sIndex)

        """ Select a feature cluster to move from, equal weight """
        # Get the number of x clusters within a y cluster with one x cluster
        # loop over feature clusters, then 1 if one unique, else 0
        x1 = [1 if len(np.unique(self._sIndex[(self._sIndex[:, 0] == i), 1])) == 1
              else 0 for i in range(self._k_counter)]
        kx1 = np.sum(x1)
        if kx1 == 0:
            logger.info('\t ... skipping Metropolis step three, no y clusters with one x cluster.')
            return
        """ first choose which feature this covariate cluster belongs from, weighted by number in that cluster. """
        cumsum = np.cumsum(x1 / kx1)
        cumbool = scipy.stats.uniform(0, 1).rvs(1) > cumsum
        j = np.min(np.where(cumbool == 0))
        # number of points in feature cluster j
        Nj = np.shape(self._sIndex[(self._sIndex[:, 0] == j), :])[0]  # np.sum(self._sIndex[:, 0] == j)
        assert Nj > 0, str(Nj)
        # points in feature cluster j
        Xj = self.x[self._sIndex[:, 0] == j, :]
        Yj = self.y[self._sIndex[:, 0] == j, :]
        assert Xj.shape[0] == Yj.shape[0] == Nj
        # print('j: {0}, Nj: {1}'.format(j, Nj))

        """ Which has only one x cluster """
        assert len(np.unique(self._sIndex[self._sIndex[:, 0] == j, 1])) == 1
        # the covariate cluster inside that cluster which will be randomly moved.
        l = np.unique(self._sIndex[self._sIndex[:, 0] == j, 1])
        # Choosing sub-cluster l in feature cluster j to propose moving
        Xlj = self.x[(self._sIndex[:, 0] == j) & (self._sIndex[:, 1] == l), :]
        #Ylj = self.y[(self._sIndex[:, 0] == j) & (self._sIndex[:, 1] == l), :]
        Nlj = np.shape(Xlj)[0]                                  # With this many samples
        assert Nlj > 0, str(Nlj)

        """ select which y cluster this can be moved to, each has equal probability. Cannot go to same """
        if self.k_unique == 1:
            logger.info('\t ... skipping Metropolis step three, no second y cluster to move to.')
            return
        # loop over unique feature co-ordinates in sIndex, keeping those != j.
        H = [i for i in np.unique(self._sIndex[:, 0]) if i != j]  # Set of feature clusters to considered moving to
        probH = np.asarray([self._expert.log_pred_marg_likelihood(self.x[self._sIndex[:, 0] == h, :],
                                                                  self.y[self._sIndex[:, 0] == h, :],
                                                                  Xj, Yj, self._theta[int(h), :]) for h in H])
        probH -= np.max(probH[probH != -np.inf])                                # exp-normalize trick
        probH = np.exp(probH) / sum(np.exp(probH))                              # normalise
        probH = np.cumsum(probH)                                                # cumulative sum of probabilities
        apbool = scipy.stats.uniform(0, 1).rvs(1) > probH                       # Choose h in H with probH
        idx = np.min(np.where(apbool == 0))
        h = int(H[int(idx)])                                                    # and continue with chosen point
        Nh = np.shape(self._sIndex[(self._sIndex[:, 0] == h), :])[0]            # number of points in feature cluster h
        assert Nh > 0, str(Nh)
        Xh = self.x[self._sIndex[:, 0] == h, :]                                 # points in feature cluster h
        Yh = self.y[self._sIndex[:, 0] == h, :]
        assert Xh.shape[0] == Yh.shape[0] == Nh
        # print('h: {0}, Nh: {1}'.format(h, Nh))

        """ Proposal state """
        # data in cluster h and from covariate cluster l, the only one in j
        Xhs = self.x[(self._sIndex[:, 0] == h) | (self._sIndex[:, 0] == j), :]
        Yhs = self.y[(self._sIndex[:, 0] == h) | (self._sIndex[:, 0] == j), :]
        assert Xhs.shape[0] == Yhs.shape[0] == Nh + Nlj
        sIndex_prop = copy.deepcopy(self._sIndex)                               # move points from lj to new cluster
        sIndex_prop[(sIndex_prop[:, 0] == j), :] = [h, self._kl_counter[h]]
        # Get the number of x clusters within a y cluster with more than one x cluster under proposal
        # loop over feature clusters, then #covariates if more than one unique, else 0
        x2ps = [len(np.unique(sIndex_prop[(sIndex_prop[:, 0] == i), 1]))
                if len(np.unique(sIndex_prop[(sIndex_prop[:, 0] == i), 1])) >= 2
                else 0 for i in range(self._k_counter + 1)]
        kx2ps = np.sum(x2ps)

        # fractions, split for notational convenience.
        LogGa_frac1 = lnGa(Nh + Nj) - lnGa(Nh) - lnGa(Nj)
        LogGa_frac2 = lnGa(self._alpha_psi[j] + Nj) + lnGa(self._alpha_psi[h] + Nh) - \
                      lnGa(self._alpha_psi[h] + Nh + Nj) - lnGa(self._alpha_psi[j])
        psi_frac = self._alpha_psi[h] / (self._alpha_psi[j] * self._alpha_theta)
        #log_likelihood_num = self._expert.log_marg_likelihood(self._theta[h, :], X=Xhs, Y=Yhs)
        likelihood_num = np.sum([np.exp(self._expert.log_pred_marg_likelihood(self.x[self._sIndex[:, 0] == htilde, :],
                                                                              self.y[self._sIndex[:, 0] == htilde, :],
                                                                              Xj, Yj, self._theta[int(htilde), :]))
                                     for htilde in np.unique(self._sIndex[:, 0]) if htilde != j])
        #log_likelihood_denom = self._expert.log_marg_likelihood(self._theta[j, :], X=Xj, Y=Yj) + \
        #                       self._expert.log_marg_likelihood(self._theta[h, :], X=Xh, Y=Yh)
        likelihood_denom = np.exp(self._expert.log_marg_likelihood(self._theta[j, :], X=Xj, Y=Yj))
        likelihood = likelihood_num / likelihood_denom
        kfrac = (kx1 * (self.k_unique-1)) / kx2ps
        #print(likelihood_num)
        #print(likelihood_denom)

        acceptance = np.exp(LogGa_frac1) * np.exp(LogGa_frac2) * psi_frac * likelihood * kfrac
        assert acceptance >= 0, \
            "negative acceptance probability: {0}, ({1},{2},{3},{4},{5})".format(acceptance, LogGa_frac1, LogGa_frac2,
                                                                                 psi_frac, likelihood, kfrac)

        if scipy.stats.uniform(0, 1).rvs(1) < acceptance:
            logger.info('\t ... accepting Metropolis step three with probability {0}'.format(acceptance))
            self._sIndex = sIndex_prop
            self._kl_counter[h] += 1
            self._theta[j, :] = np.zeros((self._theta.shape[1]))
            self._alpha_psi[j] = 0

            self.check_consistent('M3')
        else:
            logger.info('\t ... rejecting Metropolis step three with probability {0}'.format(acceptance))

    def _update_concentration(self):
        """
        update concentration (mass) parameter via auxiliary variable technique - cite Bayesian density estimation and
        inference using mixtures.
        """
        logger.info('Updating concentration parameters')

        copyAlphaTheta = self._alpha_theta
        copyPsi = copy.deepcopy(np.reshape(self._alpha_psi,(-1,1)))

        """ update alpha theta concentration parameter for feature clusters """
        phi = np.random.beta(self._alpha_theta+1, self.n)
        vthetahat = self._vtheta - np.log(phi)
        uthetahat = self._utheta + self.k_unique
        probability = (self.n*vthetahat) / (self.n*vthetahat + self._utheta +  self.k_unique - 1)
        if np.random.uniform(0, 1) < probability:
            uthetahat = uthetahat - 1
        self._alpha_theta = np.random.gamma(uthetahat, 1./vthetahat)

        """ update alpha psi concentration parameters for coviariate clusters """
        for j in range(self._k_counter+1):
            nj = sum(self._sIndex[:, 0] == j)
            if nj == 0:                                    # if empty
                pass
            else:
                kj = len(np.unique(self._sIndex[self._sIndex[:, 0] == j, 1]))
                assert kj > 0

                phij = np.random.beta(self._alpha_psi[j] + 1, nj)
                vpsijhat = self._vpsi - np.log(phij)
                upsijhat = self._upsi + kj
                probability = (nj * vpsijhat) / (nj * vpsijhat + self._upsi + kj - 1)
                if np.random.uniform(0, 1) < probability:
                    upsijhat = upsijhat - 1
                self._alpha_psi[j] = np.random.gamma(upsijhat, 1./vpsijhat)

        # Assertions
        self.check_consistent('update concentration')

        logger.debug('updated concentration: alpha_theta {0}->{1}'.format(copyAlphaTheta, self._alpha_theta))
        logger.debug('updated concentration: alpha_psi   {0}->{1}'.format(copyPsi[np.nonzero(copyPsi)],
                                                                         self._alpha_psi[np.nonzero(self._alpha_psi)]))

    def _remove_empty_(self):
        """
        Remove empty clusters
        """
        # create a copy
        effK = 0                                            # effective K, counts as we add them back in
        effKl = np.zeros(self.n+1)                          # effective Kj
        new_theta = np.zeros_like(self._theta)              # new matrix to add hyper-parameters to as we loop
        new_sIndex = np.zeros_like(self._sIndex)            # new matrix to add index allocations to as we loop
        new_alpha_psi = np.zeros_like(self._alpha_psi)      # new matrix to add mass parameters to as we loop

        for j in range(self._k_counter+1):                        # for each feature cluster

            if np.sum(self._sIndex[:, 0] == j) > 0:               # if not empty

                new_sIndex[self._sIndex[:, 0] == j, 0] = effK     # allocate to the current effective cluster.
                new_theta[effK, :] = self._theta[j, :]            # copy hyper-parameters to effective cluster.
                new_alpha_psi[effK] = self._alpha_psi[j]          # copy mass parameters to effective cluster.

                for l in range(int(self._kl_counter[j]+1)):        # for each covariate cluster in jth feature cluster.
                    if np.sum((self._sIndex[:, 0] == j) & (self._sIndex[:, 1] == l)) > 0:            # if not empty
                        new_sIndex[(self._sIndex[:, 0] == j)&(self._sIndex[:, 1] == l), 1] = effKl[effK]
                        effKl[effK] = effKl[effK]+1

                effK += 1
                assert effKl[j] >= 0, 'There must be at least one covariate cluster, {0}'.format(effKl[j])

        self._k_counter = effK
        self._kl_counter = effKl
        self._sIndex = new_sIndex
        self._theta = new_theta
        self._alpha_psi = new_alpha_psi

        # Assertions to check that copy is correct.
        as_str = 'error: ' + inspect.stack()[0][3] + ', '
        # check we have correct number of feature clusters
        assert self.k_unique == effK, as_str + 'feature clusters, {0}!={1}'.format(self.k_unique, effK)
        # check we have correct number of covariate clusters
        assert (self.kl_unique[np.nonzero(self.kl_unique)] == effKl[np.nonzero(effKl)]).all(), \
            as_str + 'covariate clusters \n{0}!=\n{1}'.format(self.kl_unique, effKl)
        assert (len(self.kl_unique[np.nonzero(self.kl_unique)]) == len(effKl[np.nonzero(effKl)])), \
            as_str + 'copy not equal.'
        self.check_consistent('remove empty')

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
        if state is None:                                       # check consistency of current state of the model
            state = self.state

        k_counter = state['k_counter']
        kl_counter = state['kl_counter']
        sIndex = state['sIndex']
        theta = state['theta']
        alpha_theta = state['alpha_theta']
        alpha_psi = state['alpha_psi']

        # counters
        assert k_counter >= max(sIndex[:, 0])
        for j in range(k_counter):
            if kl_counter[j] > 0:
                Njl = len(np.unique(sIndex[sIndex[:, 0] == j, 1]))
                assert kl_counter[j] >= Njl, \
                    'check_consistent: ' + last_update + ': Klj < Nlj: {0}<{1}'.format(kl_counter[j], Njl)
            else:
                assert kl_counter[j] == 0
        assert k_counter <= self.n

        # mass parameter for feature clusters
        assert alpha_theta > 0

        rows = np.asarray(np.unique(sIndex[:, 0]), dtype=np.int16)

        # mass parameter for covariate clusters, as these are counters, they are not necessarily reset when emptied.
        kl_filled = kl_counter[rows]
        #kl_empty = np.delete(kl_counter, rows)
        assert np.all(kl_filled > 0), \
            'check_consistent: ' + last_update + ': kl_filled not filled: {0}'.format(kl_filled)
        #assert np.all(kl_empty == 0), \
        #    'check_consistent: ' + last_update + ': kl_empty not empty: {0}'.format(kl_empty)

        # hyper-parameters
        theta_filled = theta[rows,:]
        theta_empty = np.delete(theta, rows, axis=0 )
        assert np.all(theta_filled != 0), \
            'check_consistent: ' + last_update + ': theta_filled not filled: {0}'.format(theta_filled)
        assert np.all(theta_empty == 0), \
            'check_consistent: ' + last_update + ': feature_empty not empty: {0}'.format(theta_empty)

        # _alpha_psi
        alpha_psi_filled = alpha_psi[rows]
        alpha_psi_empty = np.delete(alpha_psi, rows, axis=0)
        assert np.all(alpha_psi_filled > 0), \
            'check_consistent: ' + last_update + ': alpha_psi_filled not filled: {0}'.format(alpha_psi_filled)
        assert np.all(alpha_psi_empty == 0), \
            'check_consistent: ' + last_update + ': alpha_psi_empty not empty: {0}'.format(alpha_psi_empty)

        # check unreached indexes are empty
        assert (alpha_psi[k_counter:] == 0).all()
        assert (theta[k_counter:, :] == 0).all()

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

        assert len(self.states) > 1, '_predict_expectation: Need to update model at least once to sample GP hyper'
        assert MC > burnin, '_predict_expectation: burnin must be less than #MCMC samples. {0}!>{1}'.format(MC, burnin)
        assert x_test.ndim == 2, '_pred_expectation_state: Require 2d ndarray input x_test'
        assert x_test.shape[1] == self.xdim

        # Memory allocation
        C = np.zeros((x_test.shape[0]))                                      # Normalising constant
        if self._probit is None or return_latent is True:                    # Expectation (un-normalised)
            E = [np.zeros(x_test.shape[0]) for _ in range(self.ydim)]
        if self._probit is not None:                                         # Expectation projected
            E_projected = [np.zeros(x_test.shape[0]) for _ in range(self.ydim)]

        # Pre-computations
        hxs = np.exp(self._xlikelihood._log_full_marginal(x_test))          # Marginal likelihood

        for m in range(MC - burnin):
            pb.update_progress((m+1)/(MC - burnin), message='EmE._predict_expectation()')
            if np.mod(m, thinning):
                logger.debug('EmE._pred_expec: Thinning by skipping m: {0}'.format(m + burnin))
                continue

            # Load state
            stateM = self.states[m + burnin]
            athetam = stateM['alpha_theta']
            apsim = stateM['alpha_psi']
            thetam = stateM['theta']
            sIndexm = stateM['sIndex']
            K = stateM['k_counter']
            Kl = stateM['kl_counter']

            prior_weight = (athetam * hxs) / (athetam + self.n)
            C += prior_weight
            if self._probit is None or return_latent is True:
                E = [E[i] + prior_weight * self._expert.mu_beta(x_test, i) for i in range(self.ydim)]
            if self._probit is not None:
                E_projected = [E_projected[i] + (prior_weight *
                                                 self._probit.project_point(self._expert.mu_beta(x_test, i))[:, 0])
                               for i in range(self.ydim)]

            for j in range(K):
                njm = np.sum(sIndexm[:, 0] == j)            # number of points in cluster j for state m
                if njm > 0:                                 # if not empty
                    logging.debug('feature cluster {0}'.format(j))
                    xjm = self.x[(sIndexm[:, 0] == j), :]
                    yjm = self.y[(sIndexm[:, 0] == j), :]

                    thetajm = thetam[j, :]
                    apsijm = apsim[j]

                    term2_coef = njm / (athetam + self.n)
                    term2_brac1 = (apsijm * hxs) / (apsijm + njm)
                    term2_brac2 = 0
                    for l in range(int(Kl[j])):
                        nlj = sum((sIndexm[:, 0] == j) & (sIndexm[:, 1] == l))
                        if nlj > 0:  # if not empty
                            xlj = self.x[(sIndexm[:, 0] == j) & (sIndexm[:, 1] == l), :]
                            term2_brac2 += (nlj * np.exp(self._xlikelihood.log_predictive_marginal(x_test, xlj))) / \
                                           (apsijm + njm)

                    like_weight_j = term2_coef * (term2_brac1 + term2_brac2)
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
            assert MC <= self._mcmcSteps, 'EmE._pred_marg_expec: MC={0} > completed {1}'.format(MC, self._mcmcSteps)

        assert len(self.states) > 1, 'EmE._pred_marg_expec: Need to update model at least once to sample GP hyper'
        assert MC > burnin, 'EmE._pred_marg_expec: #burnin > #MCMC samples. {0}!>{1}'.format(MC, burnin)
        assert x_test.ndim == 2, 'EmE._pred_marg_expec: Require 2d ndarray input x_test'
        assert x_test.shape[1] == 1, 'EmE._pred_marg_expec: Test covariate dimension must be 1'
        assert isinstance(covariate, (int,)), 'EmE._pred_marg_expec: covariate must be and index integer'

        # Memory allocation
        C = np.zeros((x_test.shape[0]))                                      # Normalising constant
        if self._probit is None or return_latent is True:                    # Expectation
            E = [np.zeros(x_test.shape[0]) for _ in range(self.ydim)]
        if self._probit is not None:                                         # Expectation projected
            E_projected = [np.zeros(x_test.shape[0]) for _ in range(self.ydim)]

        # Pre-computations
        x_test_nan = np.empty((x_test.shape[0], self.xdim))
        x_test_nan[:] = np.nan
        x_test_nan[:, [covariate]] = x_test
        hxsp = np.exp(self._xlikelihood._log_full_marginal(x_test_nan[:, [covariate]]))  # Marginal likelihood

        for m in range(MC - burnin):
            pb.update_progress((m+1)/(MC - burnin), message='EmE._pred_marg_expec()')
            if np.mod(m, thinning):
                logger.debug('EmE._pred_marg_expec: Thinning by skipping m: {0}'.format(m + burnin))
                continue

            # Load state
            stateM = self.states[m + burnin]
            athetam = stateM['alpha_theta']
            apsim = stateM['alpha_psi']
            thetam = stateM['theta']
            sIndexm = stateM['sIndex']
            K = stateM['k_counter']
            Kl = stateM['kl_counter']

            pkm1 = (athetam * hxsp) / (athetam + self.n)
            C += pkm1
            if self._probit is None or return_latent is True:
                E = [E[i] + pkm1 * self._expert.mu_beta(x_test, i) for i in range(self.ydim)]
            if self._probit is not None:
                E_projected = [E_projected[i] +
                               (pkm1 * self._probit.project_point(self._expert.mu_beta(x_test, i))[:, 0])
                               for i in range(self.ydim)]

            for j in range(K):
                njm = np.sum(sIndexm[:, 0] == j)                 # number of points in cluster j for state m
                if njm > 0:                                      # if not empty
                    logging.debug('feature cluster {0}'.format(j))
                    xjm = self.x[(sIndexm[:, 0] == j), :]
                    yjm = self.y[(sIndexm[:, 0] == j), :]

                    thetajm = thetam[j, :]
                    apsijm = apsim[j]

                    pj1 = (njm / (athetam + self.n)) * (apsijm / (apsijm + njm)) * hxsp
                    C += pj1

                    # integral - approximate with importance sampling
                    # (local expectation of each cluster j, integrating out the local predictive marginal input_models)
                    if self._probit is None or return_latent is True:
                        integral = np.zeros((x_test.shape[0], 1))
                    if self._probit is not None:
                        integral_projected = np.zeros((x_test.shape[0], 1))
                    for sample in range(importance_samples):            # For each local predictive likelihood sample
                        # sample the local predictive likelihood
                        Xs = self._xlikelihood.sample_marginal(self.xdim, x_test.shape[0])
                        Xs[:, [covariate]] = x_test

                        mj, _ = self._expert.predict(xjm, yjm, Xs, thetajm)
                        if self._probit is None or return_latent is True:
                            integral += mj
                        if self._probit is not None:        # project the marginal expectation through the probit model
                            mj_projected = self._probit.project_point(mj)
                            integral_projected += mj_projected

                    if self._probit is None or return_latent is True:
                        integral /= importance_samples
                        E = [E[i] + (pj1 * integral[:, i]) for i in range(self.ydim)]
                    if self._probit is not None:
                        integral_projected /= importance_samples
                        E_projected = [E_projected[i] + (pj1 * integral_projected[:, i])
                                       for i in range(self.ydim)]

            for j in range(K):
                njm = np.sum(sIndexm[:, 0] == j)                 # number of points in cluster j for state m
                if njm > 0:                                      # if not empty
                    logging.debug('feature cluster {0}'.format(j))
                    xjm = self.x[(sIndexm[:, 0] == j), :]
                    yjm = self.y[(sIndexm[:, 0] == j), :]

                    thetajm = thetam[j, :]
                    apsijm = apsim[j]

                    for l in range(int(Kl[j])):
                        nlj = sum((sIndexm[:, 0] == j) & (sIndexm[:, 1] == l))
                        if nlj > 0:  # if not empty
                            xlj = self.x[(sIndexm[:, 0] == j) & (sIndexm[:, 1] == l), :]
                            pjl = (njm / (athetam + self.n)) * (nlj / (apsijm + njm)) * \
                                  np.exp(self._xlikelihood.log_predictive_marginal(x_test_nan, xlj, p=[covariate]))
                            C += pjl

                            # integral - approximate with importance sampling
                            # (local expectation of each cluster j, integrating out the local predictive marginal input_models)
                            if self._probit is None or return_latent is True:
                                integral = np.zeros((x_test.shape[0], 1))
                            if self._probit is not None:
                                integral_projected = np.zeros((x_test.shape[0], 1))
                            for sample in range(importance_samples):  # For each local predictive likelihood sample
                                # sample the local predictive likelihood
                                Xs = self._xlikelihood.sample_predictive_marginal(xjm, x_test.shape[0])
                                Xs[:, [covariate]] = x_test

                                mj, _ = self._expert.predict(xjm, yjm, Xs, thetajm)
                                if self._probit is None or return_latent is True:
                                    integral += mj
                                if self._probit is not None:  # project the marginal expectation through the probit model
                                    mj_projected = self._probit.project_point(mj)
                                    integral_projected += mj_projected

                            if self._probit is None or return_latent is True:
                                integral /= importance_samples
                                E = [E[i] + (pjl * integral[:, i]) for i in range(self.ydim)]
                            if self._probit is not None:
                                integral_projected /= importance_samples
                                E_projected = [E_projected[i] + (pjl * integral_projected[:, i])
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

        assert len(self.states) > 1, 'EmE._pred_dens: Need to update model at least once to sample GP hyper'
        assert MC > burnin, 'EmE._pred_dens: burnin must be less than #MCMC samples. {0}!>{1}'.format(MC, burnin)
        assert x_test.ndim == 2, 'EmE._pred_dens: Require 2d ndarray input x_test'
        assert x_test.shape[1] == self.xdim, 'EmE._pred_dens: Test covariate dim must be equal to training dim'

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
            pb.update_progress((m+1)/(MC - burnin), message='EmE._pred_dens()')

            # Thin MC chain
            if np.mod(m, thinning):
                logger.debug('EmE._pred_dens: Thinning by skipping m: {0}'.format(m + burnin))
                continue

            # Load state
            stateM = self.states[m + burnin]
            athetam = stateM['alpha_theta']
            apsim = stateM['alpha_psi']
            thetam = stateM['theta']
            sIndexm = stateM['sIndex']
            K = stateM['k_counter']
            Kl = stateM['kl_counter']

            # Prior contributions
            prior_weight = (athetam * hxs) / (athetam + self.n)
            C += prior_weight
            assert ~np.any(np.isnan(C)), f'C NaN: {C}, {prior_weight}'
            if self._probit is None or return_latent is True:
                D = [D[i] + (np.tile(prior_weight, (y_test.shape[0], 1)) * hyx_marg) for i in range(self.ydim)]
                assert ~np.any(np.isnan(D)), f'D NaN: {D}'
            if self._probit is not None:
                D_projected = [D_projected[i] + (np.tile(prior_weight, (len(self._probit.categories), 1)) *
                                                 hyx_marg_projected) for i in range(self.ydim)]

            # Posterior contributions
            for j in range(K):
                 njm = np.sum(sIndexm[:, 0] == j)
                 if njm > 0:  # if not empty
                     logging.debug('feature cluster {0}'.format(j))
                     xjm = self.x[(sIndexm[:, 0] == j), :]
                     yjm = self.y[(sIndexm[:, 0] == j), :]

                     thetajm = thetam[j, :]
                     apsijm = apsim[j]

                     term2_coef = njm / (athetam + self.n)
                     term2_brac1 = (apsijm * hxs) / (apsijm + njm)
                     term2_brac2 = 0
                     for l in range(int(Kl[j])):
                         nlj = sum((sIndexm[:, 0] == j) & (sIndexm[:, 1] == l))
                         # if not empty
                         if nlj > 0:
                             xlj = self.x[(sIndexm[:, 0] == j) & (sIndexm[:, 1] == l), :]
                             term2_brac2 += (nlj * np.exp(self._xlikelihood.log_predictive_marginal(x_test, xlj))) /\
                                            (apsijm + njm)

                     like_weight_j = term2_coef * (term2_brac1 + term2_brac2)
                     C += like_weight_j
                     assert ~np.any(np.isnan(C)), f'C NaN: {C}, {like_weight_j}'

                     mj, sj = self._expert.predict(xjm, yjm, x_test, thetajm)
                     if self._probit is None or return_latent is True:
                         hyxL = np.transpose(scipy.stats.norm.pdf(y_test, mj, sj))
                         D = [D[i] + (np.tile(like_weight_j, (y_test.shape[0], 1)) * hyxL) for i in range(self.ydim)]
                         assert ~np.any(np.isnan(D)), f'D NaN: {D}, {like_weight_j}, {hyxL}'
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
            assert MC <= self._mcmcSteps, '_predict_density: MC={0} > completed {1}'.format(MC, self._mcmcSteps)

        assert len(self.states) > 1, 'EmE._pred_marg_dens: Need to update model at least once to sample GP hyper'
        assert MC > burnin, 'EmE._pred_marg_dens: #burnin > #MCMC samples. {0}!>{1}'.format(burnin, MC)
        assert x_test.ndim == 2, 'EmE._pred_marg_dens: Require 2d ndarray input x_test'
        assert x_test.shape[1] == 1, 'EmE._pred_marg_dens: Test covariate dimension must be 1'
        assert isinstance(covariate, (int,)), 'EmE._pred_marg_dens: Covariate must be and index integer'

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
        hxsp = np.exp(self._xlikelihood._log_full_marginal(x_test_nan[:, [covariate]]))           # Marginal likelihood
        if self._probit is None or return_latent is True:                   # Approximate marginal likelihood (response)
            hyx_marg = self._expert.prior_marg_likelihood(x_test, y_test, samples=50)
        if self._probit is not None:
            hyx_marg_projected = self._expert.log_prior_marg_probit(x_test, self._probit, samples=50)

        for m in range(MC - burnin):
            pb.update_progress((m+1)/(MC - burnin), message='EmE._predict_marginal_density()')
            if np.mod(m, thinning):
                logger.debug('EmE._pred_marg_dens: Thinning by skipping m: {0}'.format(m + burnin))
                continue

            # Load state
            stateM = self.states[m + burnin]
            athetam = stateM['alpha_theta']
            apsim = stateM['alpha_psi']
            thetam = stateM['theta']
            sIndexm = stateM['sIndex']
            K = stateM['k_counter']
            Kl = stateM['kl_counter']

            # prior contributions
            pkm1 = (athetam * hxsp) / (athetam + self.n)
            C += pkm1
            if self._probit is None or return_latent is True:
                D = [D[i] + (np.tile(pkm1, (y_test.shape[0], 1)) * hyx_marg) for i in range(self.ydim)]
            if self._probit is not None:
                D_projected = [D_projected[i] + (np.tile(pkm1, (len(self._probit.categories), 1)) *
                                                 hyx_marg_projected) for i in range(self.ydim)]

            # posterior contributions
            for j in range(K):
                 njm = np.sum(sIndexm[:, 0] == j)
                 if njm > 0:  # if not empty
                     logging.debug('feature cluster {0}'.format(j))
                     xjm = self.x[(sIndexm[:, 0] == j), :]
                     yjm = self.y[(sIndexm[:, 0] == j), :]

                     thetajm = thetam[j, :]
                     apsijm = apsim[j]

                     pj1 = (njm / (athetam + self.n)) * (apsijm / (apsijm + njm)) * hxsp
                     C += pj1

                     # integral - approximate with importance sampling
                     # (local expectation of each cluster j, integrating out the local predictive marginal input_models)
                     if self._probit is None or return_latent is True:
                        integral = np.zeros((y_test.shape[0], x_test.shape[0]))
                     if self._probit is not None:
                        integral_projected = np.zeros((len(self._probit.categories), x_test.shape[0]))
                     for sample in range(importance_samples):  # for each local predictive likelihood sample
                         # sample the local predictive likelihood
                         Xs = self._xlikelihood.sample_marginal(self.xdim, x_test.shape[0])
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
                         D = [D[i] + (np.tile(pj1, (y_test.shape[0], 1)) * integral) for i in range(self.ydim)]
                     if self._probit is not None:
                         integral_projected /= importance_samples
                         D_projected = [D_projected[i] +
                                        (np.tile(pj1, (len(self._probit.categories), 1)) * integral_projected)
                                        for i in range(self.ydim)]

            for j in range(K):
                njm = np.sum(sIndexm[:, 0] == j)  # number of points in cluster j for state m
                if njm > 0:  # if not empty
                    logging.debug('feature cluster {0}'.format(j))
                    xjm = self.x[(sIndexm[:, 0] == j), :]
                    yjm = self.y[(sIndexm[:, 0] == j), :]

                    thetajm = thetam[j, :]
                    apsijm = apsim[j]

                    for l in range(int(Kl[j])):
                        nlj = sum((sIndexm[:, 0] == j) & (sIndexm[:, 1] == l))
                        if nlj > 0:  # if not empty
                            xlj = self.x[(sIndexm[:, 0] == j) & (sIndexm[:, 1] == l), :]
                            pjl = (njm / (athetam + self.n)) * (nlj / (apsijm + njm)) * \
                                  np.exp(self._xlikelihood.log_predictive_marginal(x_test_nan, xlj, p=[covariate]))
                            C += pjl

                            # integral - approximate with importance sampling
                            # (local expectation of each cluster j, integrating out the local predictive marginal input_models)
                            if self._probit is None or return_latent is True:
                                integral = np.zeros((y_test.shape[0], x_test.shape[0]))
                            if self._probit is not None:
                                integral_projected = np.zeros((len(self._probit.categories), x_test.shape[0]))
                            for sample in range(importance_samples):  # For each local predictive likelihood sample
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
                                D = [D[i] + (np.tile(pjl, (y_test.shape[0], 1)) * integral) for i in range(self.ydim)]
                            if self._probit is not None:
                                integral_projected /= importance_samples
                                D_projected = [D_projected[i] +
                                               (np.tile(pjl, (len(self._probit.categories), 1)) * integral_projected)
                                               for i in range(self.ydim)]

        returnDict = dict()
        returnDict['Xgrid'] = Xgrid
        returnDict['Ygrid'] = Ygrid
        if 'D' in locals():
            returnDict['D'] = [D[i] / np.tile(C, (y_test.shape[0], 1)) for i in range(self.ydim)]
        if 'D_projected' in locals():
            returnDict['Dp'] = [D_projected[i] / np.tile(C, (len(self._probit.categories), 1))
                                for i in range(self.ydim)]

        return returnDict

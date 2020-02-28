"""
A class for Gaussian process experts with a squared exponential kernel, where each feature is assumed independent
(product). We can optionally have a zero-mean, linear-mean, or constant-mean GP.

Author:
    Charles Gadd

Date:
    27/02/2018
"""
from __future__ import division
import GPy
import MixtureOfExperts.utils.inference as inference
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
import copy
import sys

import logging
logger = logging.getLogger(__name__)

gplogger = logging.getLogger("GP")
gplogger.setLevel(logging.WARNING)

__all__ = ['IndpendentRBF']


class IndpendentRBF(object):

    # public (accessible through @property decorators below)
    _input_dim = None
    _output_dim = None
    _n_hyper = None                 # number of expert hyper-parameters for memory allocation purposes.
    _n_hyper_indGP = None           # number of independent GP hyper-parameters for memory allocation purposes.
    _process_mean = None            # String or False, indicating mean process.
    _prior_mean = None              # prior for the process mean parameters, if not False.
    _prior_nugget = None            # prior parameters for sigma_j^2
    _prior_lengthscale = None       # prior on lengthscales
    _prior_signal = None            # prior parameters \nu_j for the P input kernels, equal across P inputs
    _index_nugget = None
    _index_lengthscale = None
    _index_signal = None
    _index_mean = None
    # private
    _parameter_names = None         # list of strings for parameters names in an independent GP (length _n_hyper_indGP)
    _ARD = None

    @property
    def num_covariates(self):
        """ Get the number of covariates. """
        return self._input_dim

    @property
    def num_features(self):
        """ Get the number of features. """
        return self._output_dim

    @property
    def n_hyper(self):
        """ Get the number of hyper-parameters """
        return self._n_hyper

    @property
    def kernel_variance(self):
        """ Get the signal variance prior of the kernel """
        return self._prior_signal

    @kernel_variance.setter
    def kernel_variance(self, prior):
        self._prior_signal = prior

    @property
    def lengthscale(self):
        """ Get the lengthscale prior """
        return self._prior_lengthscale

    @lengthscale.setter
    def lengthscale(self, prior):
        """ Set the lengthscale priors. Give a list of priors (ARD=True), or a single prior (ARD==True and shared) or
        (ARD==False)"""
        if type(prior) is list:
            assert self._ARD is True, 'Give a list of lengthscale priors only when using an ARD kernel'
            assert len(prior) == self._input_dim, 'Give every lengthscale prior in the list, or pass a shared prior'
            self._prior_lengthscale = prior
        elif self._ARD is True:
            self._prior_lengthscale = [prior for _ in range(self._input_dim)]
        else:
            self._prior_lengthscale = prior

    @property
    def likelihood_variance(self):
        """ Get the likelihood variance prior (nugget) """
        return self._prior_nugget

    @likelihood_variance.setter
    def likelihood_variance(self, prior):
        self._prior_nugget = prior

    @property
    def hyper_priors(self):
        return self.lengthscale, self.kernel_variance, self.likelihood_variance

    @property
    def kern(self):
        """ Get the kernel object. """
        return GPy.kern.RBF(input_dim=self._input_dim, ARD=self._ARD)

    @property
    def process_mean(self):
        if self._process_mean is not False:
            if self._process_mean == 'Linear':
                return GPy.mappings.Linear(self._input_dim, 1)
            elif self._process_mean == 'Constant':
                return GPy.mappings.Constant(self._input_dim, 1)
            else:
                raise ValueError('Bad mean process')
        else:
            return None

    @property
    def param_names(self):
        """ Get the parameter names inside an independent GP, used internally """
        if self._parameter_names is not None:
            return self._parameter_names
        else:
            # build the model with arbitrary data to extract parameter list under our dimensions and mean choice
            gp = GPy.models.GPRegression(np.zeros((1, self._input_dim)), np.ones((1, self._output_dim)),
                                               kernel=self.kern, mean_function=self.process_mean)
            self._parameter_names = gp.parameter_names_flat()
            return self._parameter_names

    @property
    def index_nugget(self):
        """ Get the indexes for the nugget in theta """
        if self._index_nugget is None:
            names = self.param_names
            self._index_nugget = np.asarray([i for i, s in enumerate(names) if 'Gaussian_noise.variance' in s])
        return self._index_nugget

    @property
    def index_lengthscale(self):
        """ Get the indexes for the lengthscales in theta """
        if self._index_lengthscale is None:
            names = self.param_names
            self._index_lengthscale = np.asarray([i for i, s in enumerate(names) if 'rbf.lengthscale' in s])
        return self._index_lengthscale

    @property
    def index_signal(self):
        """ Get the indexes for the kernel variances in theta """
        if self._index_signal is None:
            names = self.param_names
            self._index_signal = np.asarray([i for i, s in enumerate(names) if 'rbf.variance' in s])
        return self._index_signal

    @property
    def index_mean(self):
        """ Get the indexes of the mean process hyper-parameters in theta """
        if self._index_mean is None:
            names = self.param_names
            self._index_mean = np.asarray([i for i, s in enumerate(names) if ('linmap.A' in s) or ('constmap.C' in s)])
        return self._index_mean

    def __init__(self, input_dim, output_dim, process_mean=False, ard=False, name=__name__):
        """
        Initialise class for independent Gaussian process experts across features. We have the option of zero, constant,
        or linearly meaned process.

        :param input_dim:               The number of covariates in our model
        :param output_dim:              The number of features in our model
        :param process_mean:            String ('Constant', 'Linear') or False (zero mean), giving the mean process.
        :param name:                    The class name
        """
        self.__name__ = name
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._ARD = ard

        # Default hyper-priors, shared across P. These can be changed with setters above.
        self.kernel_variance = GPy.priors.Gamma(2., 10.)                 # signal variance, nu_j,    2,4
        self.lengthscale = GPy.priors.LogGaussian(np.log(0.25), 1. / 4)  # length scales, l_j        0.3,...
        self.likelihood_variance = GPy.priors.Gamma(2, 40.)              # noise, sigma_j^2          1.1,40

        # add optional default hyper-priors and update number of hyper-parameters
        self._process_mean = process_mean                             # Set string
        if process_mean is not False:
            if process_mean == 'Linear':                              # If linear mean process, m(x)=AX
                self._prior_mean = GPy.priors.Gaussian(0, 0.1)        # Mean process parameter prior, on A
            elif process_mean == 'Constant':                          # If constant mean process m(x) = C
                self._prior_mean = GPy.priors.Gaussian(15, 5)         # Mean process parameter prior, on C
            else:
                raise ValueError('Bad mean process')

        self._n_hyper_indGP = len(self.param_names)                   # number of hypers in zero-mean, updated if not
        self._n_hyper = self._n_hyper_indGP * self._output_dim        # Number of expert hyper-parameters, all features

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = "EXPERT: " + self.__name__ + '\n'
        s += "\t Number of covariate dimensions: {0} \n".format(str(self.num_covariates))
        s += "\t Number of features: {0} \n".format(str(self.num_features))
        if self._process_mean is not False:
            s += "\t Mean process: " + str(self.process_mean.name) + '\n'
        s += "\t Kernel: " + self.kern.name + '\n'
        s += "\t Number of hyper-parameters: {0}".format(str(self._n_hyper)) + '\n'
        s += "\t Priors: \n"
        s += "\t\t kernel variance " + str(self.kernel_variance) + '\n'
        s += "\t\t likelihood variance " + str(self.likelihood_variance) + '\n'
        if type(self.lengthscale) is list:
            for i, lenprior in enumerate(self.lengthscale):
                s += '\t\t lengthscale {0} '.format(i) + str(lenprior) + '\n'
        else:
            s += '\t\t lengthscale ' + str(self.lengthscale) + '\n'
        if self._process_mean is not False:
            s += '\t\t mean process ' + str(self._prior_mean) + '\n'

        return s

    def log_marg_likelihood(self, theta, X, Y):
        """
        Marginal likelihood for Y conditioned on X.

        :param X:              Observed features
           :type               np.ndarray (#samples * #covariates)
        :param Y:              Observed covariates
           :type               np.ndarray (#samples * #features)
        :param theta:          Gaussian process hyper-parameters our functional is conditioned on. Vectorised.
           :type               np.ndarray (#hyper-parameters * 1)
        """
        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        if Y.ndim == 1:
            Y = np.reshape(Y, (-1, 1))

        assert X.ndim == 2 and Y.ndim == 2, \
            'log_marg_likelihood: not 2d ndarrays. ({0} & {1})!=2'.format(X.ndim, Y.ndim)
        assert X.shape[0] == Y.shape[0], \
            'log_marg_likelihood: bad sample size. {0} != {1}'.format(X.shape[0], Y.shape[0])
        assert len(theta) == self._n_hyper,\
            'log_marg_likelihood: bad theta size. ({0},{1})'.format(theta.shape[0], theta.shape[1])

        log_marginal_likelihood = 0
        for feature in range(self._output_dim):
            # create model
            yk = np.reshape(Y[:, feature], (-1, 1))
            gp_local = GPy.models.GPRegression(X, yk, kernel=self.kern, mean_function=self.process_mean)
            # add hyper-parameters back into model
            itershift = feature * self._n_hyper_indGP
            gp_local.param_array[:] = theta[itershift:itershift+self._n_hyper_indGP]
            gp_local._trigger_params_changed()

            gp_local_lik = gp_local.log_likelihood()
            if ~np.isinf(gp_local_lik) and ~np.isnan(gp_local_lik):
                log_marginal_likelihood += gp_local_lik
            else:
                print('warning - log marginal likelihood {0} setting equal to -inf'.format(gp_local_lik))
                print(theta)
                log_marginal_likelihood = -np.inf

        assert np.exp(log_marginal_likelihood) >= 0, 'non-positive probability {0}'.format(log_marginal_likelihood)

        return log_marginal_likelihood

    def log_prior_marg(self, xstar, ystar, samples=300):
        logging.warning('log_prior_marg deprecated, use prior_marg_likelihood')
        return self.prior_marg_likelihood(xstar, ystar, samples)

    def prior_marg_likelihood(self, xstar, ystar, samples=300):
        """
        Calculate the marginal likelihood (marginalised over latent functions and hyper-parameters) at test points.
        """
        hyx_marg2 = np.zeros((ystar.shape[0], xstar.shape[0]))  # latent and hyper marginalised
        for col in range(xstar.shape[0]):

            hyx_marg2[:, col] = np.mean([norm.pdf(ystar,
                                              self.mu_beta(np.reshape(xstar[col, 0], (1, 1)), 0,
                                                                            samples=1), # TODO: generalise covariates
                                               np.sqrt(self.kernel_variance.rvs(1) + self.likelihood_variance.rvs(1)))
                                         for _ in range(samples)], axis=0)

        return hyx_marg2

    def log_prior_marg_probit(self, xstar, probit, samples=300):
        logging.warning('log_prior_marg_probit deprecated, use prior_marg_likelihood_probit')
        return self.prior_marg_likelihood_probit(xstar, probit, samples)

    def prior_marg_likelihood_probit(self, xstar, probit, samples=300):
        """
        Calculate the predictive marginal likelihood (marginalised over latent functions and hyper-parameters),
        projecting each sample through the probit model.
        """
        hyx_marg2 = np.zeros((len(probit.categories), xstar.shape[0]))  # latent and hyper marginalised
        for col in range(xstar.shape[0]):
            hyx_marg2[:, col] = np.mean([probit.project_density(
                                                self.mu_beta(np.reshape(xstar[col, 0], (1, 1)), 0, samples=1), # TODO: generalise covariates
                                                np.sqrt(self.kernel_variance.rvs(1) + self.likelihood_variance.rvs(1)))
                                        for _ in range(samples)], axis=0).reshape((-1,))

        return hyx_marg2

    def log_pred_marg_likelihood(self, X, Y, xstar, ystar, theta):
        """
        Predictive likelihood for ystar given xstar, conditioned on X and Y.

        :param X:              Observed features
           :type               np.ndarray (#samples * #covariates)
        :param Y:              Observed covariates
           :type               np.ndarray (#samples * #features)
        :param xstar:          Covariates we predict for
           :type               np.ndarray (#samples * #covariates)
        :param ystar:          Features whose density we wish to obtain
           :type               np.ndarray (#samples * #features)       
        :param theta:          Gaussian process hyper-parameters our functional is conditioned on. Vectorised.
           :type               np.ndarray (#hyper-parameters * 1)                                  
        """

        gploglikxi = 0

        if X.ndim == 1:
            X = np.reshape(X, (-1,1))
        if Y.ndim == 1:
            Y = np.reshape(Y, (-1,1))

        assert X.ndim == 2 and Y.ndim == 2, '_init: require 2d ndarrays. ({0} & {1})!=2'.format(X.ndim, Y.ndim)
        assert X.shape[0] == Y.shape[0], '_init: inconsistent sample size. {0} != {1}'.format(X.shape[0], Y.shape[0])
        assert len(theta) == self._n_hyper * self._output_dim

        """build model"""
        for feature in range(self._output_dim):
            # create model
            yk = np.reshape(Y[:, feature], (-1, 1))
            gp_local = GPy.models.GPRegression(X, yk, kernel=self.kern, mean_function=self.process_mean)

            # add hyper-parameters back into model
            itershift = feature * self._n_hyper_indGP
            gp_local.param_array[:] = theta[itershift:itershift+self._n_hyper_indGP]
            gp_local._trigger_params_changed()              # not sure if this is needed, call incase

            # predictive distribution using model
            p = gp_local.predict(xstar, full_cov=True)
            gp_local_logpred = mvn.logpdf(ystar[:, feature], p[0].flatten(), p[1])      # gp pred lik for each feature

            # Assume independence (for debugging)
            #p_debug = gp_local.predict(xstar, full_cov=False)
            #gp_local_logpred_debug = np.sum(norm.logpdf(ystar[:, feature],
            #                                            p_debug[0].flatten(),
            #                                            np.sqrt(p_debug[1]).flatten()))
            #if ystar.shape[0] > 1:
            #    print('a')
            #    print(f'gp pred likelihood for independent feature: {gp_local_logpred}')
            #    print(f'gp pred likelihood for independent feature: {gp_local_logpred_debug}')
            #    import matplotlib.pyplot as plt
            #    plt.imshow(p[1])
            #    plt.colorbar()
            #    plt.show()
            #else:
            #    assert np.isclose(gp_local_logpred, gp_local_logpred_debug)

            if ~np.any(np.isnan(gp_local_logpred)) and ~np.any(np.isinf(gp_local_logpred)):
                gploglikxi += gp_local_logpred
            else:
                print('warning - log predictive likelihood {0} setting equal to -inf'.format(gp_local_logpred))
                gploglikxi = -np.inf

            assert ~np.any(np.isnan(np.exp(gploglikxi))) and ~np.any(np.isinf(np.exp(gploglikxi))), \
                'error in GP predictive likelihood: {0}, {1}, {2}, {3}'.format(gploglikxi, ystar[:, feature], p[0],
                                                                               np.sqrt(p[1]))

        return gploglikxi

    def predict(self, X, Y, xstar, theta):
        """
        Predict the moments of a feature at test inputs, given a parameterisation

        :param X:              Observed features
           :type               np.ndarray (#samples * #covariates)
        :param Y:              Observed covariates
           :type               np.ndarray (#samples * #features)
        :param xstar:          Covariates we predict for
           :type               np.ndarray (#samples * #covariates)
        :param theta:          Gaussian process hyper-parameters our functional is conditioned on. Vectorised.
           :type               np.ndarray (#hyper-parameters * 1)
        """

        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        if Y.ndim == 1:
            Y = np.reshape(Y, (-1, 1))

        assert X.ndim == 2 and Y.ndim == 2, 'predict: require 2d ndarrays. ({0} & {1})!=2'.format(X.ndim, Y.ndim)
        assert X.shape[0] == Y.shape[0], 'predict: inconsistent sample size. {0} != {1}'.format(X.shape[0], Y.shape[0])

        means = np.zeros((xstar.shape[0], self._output_dim))
        std = np.zeros((xstar.shape[0], self._output_dim))

        """build model"""
        for feature in range(self._output_dim):
            # create model
            yk = np.reshape(Y[:, feature], (-1, 1))
            gp_local = GPy.models.GPRegression(X, yk, kernel=self.kern, mean_function=self.process_mean)

            # add hyper-parameters back into model
            itershift = feature * self._n_hyper_indGP
            gp_local.param_array[:] = theta[itershift:itershift + self._n_hyper_indGP]
            gp_local._trigger_params_changed()  # not sure if this is needed, call in case

            # predictive distribution using model
            p = gp_local.predict(xstar, full_cov=False)
            means[:, feature] = p[0][:, 0]
            std[:, feature] = np.sqrt(p[1][:, 0])

        return means, std

    def mu_beta(self, X, feature=0, samples=200):
        """
        Get the prior mean at input X for a given set of hyper-parameters
        TODO: GPy.prior does not have a function for analytically obtaining prior mean, so currently approximate.
        TODO: check this works for non-Constant means, should be ok
        TODO: If you use this for priors which depend on the covariate dimension (linear) then this needs to be modifed
              (you need to specify the covariate for when this is the case - for use in marginalised predictions)

        :param X:
        :param theta:
        :return:
        """
        assert X.ndim == 2, 'mu_beta: X ndim {0}'.format(X.ndim)
        #assert X.shape[1] == self.num_covariates, 'mu_beta: x shape ({0})'.format(X.shape)
        assert self._output_dim == 1, 'This has not been written for multi-dimensional output' #TODO: multi output
        assert feature == 0, 'This has not been written for multi-dimensional output'          #TODO: multi output

        mean_function = self.process_mean
        mu_beta = np.zeros((X.shape[0],))

        if self._process_mean == 'Linear':
            raise NotImplementedError('Need to allow correct covariate dependency')
            # for _ in range(samples):
            #     hyper_sample = self._prior_mean.rvs(len(self.index_mean))
            #     mean_function.A = hyper_sample
            #     mu_beta += mean_function.f(X)[:, 0]
            # mu_beta /= samples
        elif self._process_mean == 'Constant':
            for _ in range(samples):
                hyper_sample = self._prior_mean.rvs(len(self.index_mean))
                mean_function.C = hyper_sample
                mu_beta += mean_function.f(X)[:, 0]
            mu_beta /= samples
        else:
            pass

        return mu_beta

    def prior(self, samples):
        """ Obtain samples of the hyper-prior, and return with same index ordering as used throughout """

        theta = np.ones((samples, self._n_hyper))
        names = self.param_names

        for feature in range(self._output_dim):
            itershift = feature * self._n_hyper_indGP

            p = 0
            for hyperparam in range(self._n_hyper_indGP):
                if 'rbf.lengthscale' in names[hyperparam]:
                    if self._ARD is True:
                        theta[:, hyperparam + itershift] = self.lengthscale[p].rvs(samples)
                        p += 1
                    else:
                        theta[:, hyperparam + itershift] = self.lengthscale.rvs(samples)
                elif 'rbf.variance' in names[hyperparam]:
                    theta[:, hyperparam + itershift] = self.kernel_variance.rvs(samples)
                elif 'Gaussian_noise.variance' in names[hyperparam]:
                    theta[:, hyperparam + itershift] = self.likelihood_variance.rvs(samples)
                elif ('linmap.A' in names[hyperparam]) or ('constmap.C' in names[hyperparam]):
                    theta[:, hyperparam + itershift] = self._prior_mean.rvs(samples)
                else:
                    logger.warning('Unable to set prior of hyper-parameter')
                    raise ValueError

        return theta

    def posterior(self, X, Y, init=False, samples=150, stepsize=0.15, stepsize_range=[1e-6, 1e-1]):
        """
        Sample the hyper-parameter posterior of expert.
        
        :param X:              Observed features
           :type                np.ndarray (#samples * #covariates)
        :param Y:              Observed covariates
           :type                np.ndarray (#samples * #features) 
        :param stepsize        stepsize for HMC
            :type               float
        :param hmcsamples      Number of Hamiltonian Monte Carlo samples (including burn in period) when sampling 
                                posterior
            :type               float
        :param hmcinit         How to initialise the MC for HMC
            :type                bool (True = approximate prior mean, False = default)
                                 np.ndarray (the vector of parameters to start from)
        :return:               Vector of posterior samples.
        """

        theta = np.zeros(self._n_hyper)
        chain = []
        for feature in range(self._output_dim):
            # get data/indexes specific to feature
            ykf = np.reshape(Y[:, feature], (-1, 1))
            itershift = feature * self._n_hyper_indGP

            # Create GP model
            gp_local = GPy.models.GPRegression(X, ykf, kernel=self.kern, mean_function=self.process_mean)

            # set priors
            gp_local.kern.variance.set_prior(self.kernel_variance, warning=False)
            if self._ARD is True:
                for p in range(self._input_dim):
                    gp_local.kern.lengthscale[[p]].set_prior(self.lengthscale[p], warning=False)
            else:
                gp_local.kern.lengthscale.set_prior(self.lengthscale, warning=False)
            gp_local.likelihood.variance.set_prior(self.likelihood_variance, warning=False)  # nugget
            if self._process_mean is not False:
                if self._process_mean == 'Linear':
                    gp_local.mean_function.A.set_prior(self._prior_mean, warning=False)
                    logging.debug('setting linear A prior')
                elif self._process_mean == 'Constant':
                    logging.debug('setting constant C prior')
                    gp_local.mean_function.C.set_prior(self._prior_mean, warning=False)

            # Initialise HMC in roughly the correct region (how depends on hmcinit)
            if isinstance(init, bool):
                if init is True:
                    logging.debug('Initialising with MAP estimates')
                    #print(gp_local.kern.lengthscale)
                    #print(gp_local.likelihood.variance)
                    gp_local.optimize()
                    #print(gp_local.kern.lengthscale)
                    #print(gp_local.likelihood.variance)
                elif init is False:
                    logging.debug('Using prior to initialise HMC')
                    gp_local.kern.variance = np.mean(self.kernel_variance.rvs(100))
                    if self._ARD is True:
                        for p in range(self._input_dim):
                            gp_local.kern.lengthscale[[p]] = np.mean(self.lengthscale[p].rvs(100))
                    else:
                        gp_local.kern.lengthscale = np.mean(self.lengthscale.rvs(100))
                    gp_local.likelihood.variance = np.mean(self.likelihood_variance.rvs(100))
                    if self._process_mean is not False:
                        if self._process_mean == 'Linear':
                            gp_local.mean_function.A = np.mean(self._prior_mean.rvs(100))
                        elif self._process_mean == 'Constant':
                            gp_local.mean_function.C = np.mean(self._prior_mean.rvs(100))
            elif len(init) == self.n_hyper:
                logging.debug('Using specified initialisation of HMC')
                gp_local.kern.variance = init[self.index_signal]
                if self._ARD is True:
                    for p in range(self._input_dim):
                        gp_local.kern.lengthscale[[p]] = init[self.index_lengthscale[p]]
                else:
                    gp_local.kern.lengthscale = init[self.index_lengthscale]
                gp_local.likelihood.variance = init[self.index_nugget]
                if self._process_mean is not False:
                    if self._process_mean == 'Linear':
                        gp_local.mean_function.A = init[self.index_mean]
                    elif self._process_mean == 'Constant':
                        gp_local.mean_function.C = init[self.index_mean]
            else:
                logging.critical(f'Unable to intialise HMC with initialisation {init}')
                sys.exit()




            # Sample GP hyperparameters, order: mean, signal, lengthscale, nugget
            gp_copy = copy.deepcopy(gp_local)
            #shmc_local = inference.metropoliswrapper(gp_local, Ntotal=5000, Nburn=1000, Nthin=1, tune_throughout=True)
            shmc_local = inference.hmcwrapper(gp_local, hmcsamples=samples, stepsize=stepsize)
            #shmc_local = inference.hmcshortcutwrapper(gp_local, hmcsamples=samples, stepsize_range=stepsize_range )     #TODO: try to make stepsize dependent on hyper-parameters scale

            # plot posterior (debugging)
            #import matplotlib.pyplot as plt
            #for iter in range(shmc_local.shape[1]):
            #    plt.subplot(211)
            #    plt.plot(np.linspace(1, shmc_local.shape[0] - 1, shmc_local.shape[0]), shmc_local[:, iter])
            #    plt.subplot(212)
            #    plt.hist(shmc_local[:, iter])
            #    plt.show()

            try:
                if np.any(np.var(shmc_local, axis=0) == 0) or np.any(shmc_local[-1, :] == shmc_local[0, :]):
                    debugging_plot = False
                    if debugging_plot:
                        import matplotlib.pyplot as plt
                        # debugging plots
                        for iter in range(shmc_local.shape[1]):
                            plt.subplot(211)
                            plt.plot(np.linspace(1, shmc_local.shape[0] - 1, shmc_local.shape[0]), shmc_local[:, iter])
                            plt.subplot(212)
                            plt.hist(shmc_local[:, iter])
                            plt.show()
                        raise ValueError('HMC not mixing - refer to plot...')
            except ValueError:
                np.set_printoptions(threshold=np.nan)
                report_string = 'Handling demo-time error' +\
                                '\n posterior variances {0}'.format(np.var(shmc_local, axis=0)) + \
                                '\n start mean: {0}'.format(gp_copy.param_array[self.index_mean]) + \
                                '\n end mean: {0}'.format(gp_local.param_array[self.index_mean]) + \
                                '\n start signal: {0}'.format(gp_copy.param_array[self.index_signal]) + \
                                '\n end signal: {0}'.format(gp_local.param_array[self.index_signal]) + \
                                '\n start lengthscale: {0}'.format(gp_copy.param_array[self.index_lengthscale]) + \
                                '\n end lengthscale: {0}'.format(gp_local.param_array[self.index_lengthscale]) + \
                                '\n start nugget: {0}'.format(gp_copy.param_array[self.index_nugget]) + \
                                '\n end nugget: {0}'.format(gp_local.param_array[self.index_nugget]) + \
                                '\n Input shape: {0}'.format(gp_local.X.shape) + \
                                '\n Kernel: {0}'.format(gp_local.kern) + \
                                '\n Kernel diagonal: {0}'.format(np.diagonal(gp_local.kern.K(gp_local.X), 1)) + \
                                '\n Y meta data: {0}'.format(gp_local.Y_metadata) + \
                                '\n nugget: {0}'.format(gp_local.likelihood.gaussian_variance(gp_local.Y_metadata))
                logging.exception(report_string)
                raise

            chain.append(shmc_local)

            # Store last step of hmc
            theta[itershift: self._n_hyper_indGP + itershift] = shmc_local[-1, :]

        return theta, chain

    def plot_hyperpriors(self, separate=False, logxscale=False, names=False):
        """
        Plot the hyper-priors
        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError('cannot import matplotlib')

        print(self.param_names)

        for i in self.hyper_priors:
            if type(i) is list:
                for j in range(len(i)):
                    i[j].plot()
                    if logxscale:
                        plt.xscale('log')
                    if separate:
                        plt.show()

            else:
                i.plot()
                if logxscale:
                    plt.xscale('log')
            plt.show()

        if self._process_mean is not False:
            self._prior_mean.plot()
            plt.show()

    def plot_1d(self, x, y, theta, feature):

        gp_local = GPy.models.GPRegression(x, y, kernel=self.kern, mean_function=self.process_mean)

        # add hyper-parameters back into model
        itershift = feature * self._n_hyper_indGP
        gp_local.param_array[:] = theta[itershift:itershift + self._n_hyper_indGP]
        gp_local._trigger_params_changed()  # not sure if this is needed, call in case

        gp_local.plot()

    # backwards compatibility
    def log_pred_density(self, X, Y, xstar, ystar, theta):
        logging.warning('This function will be renamed')
        return self.log_pred_marg_likelihood(X, Y, xstar, ystar, theta)

    def expected_prior_process_mean(self, X, feature=0, samples=200):
        logging.warning('expected_prior_process_mean, use mubeta')
        return self.mu_beta(X, feature=feature, samples=samples)
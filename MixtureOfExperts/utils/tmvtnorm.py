from __future__ import division
import numpy as np
import logging
logger = logging.getLogger(__name__)

__all__ = ['tmvtnorm_sample', 'tmvtnorm_rpy2', 'tmvtnorm_emcee']


def importr_tryhard(packname):
    """
    Load R package. If package cannot be loaded then install. If you have problems with this function, try making sure
    that the default respository contains the package.

    :param packname:    name of package to install
    :return:

    TODO: R libraries currently get re-loaded every time, need to change so its done once on initial imports.
    """
    logger.debug(f'Loading R package {packname}')

    import rpy2.robjects.packages as rpackages
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=2)                            # select first mirror in the list
    from rpy2.robjects.vectors import StrVector

    try:
        rpack = rpackages.importr(packname)
    except:
        try:
            #print(utils.install_packages.)
            utils.install_packages(StrVector(packname), repos='http://cran.us.r-project.org')
            rpack = rpackages.importr(packname)
        except:
            raise RuntimeError(f'Unable to install {packname}')

    return rpack


def tmvtnorm_sample(samples, mean, sigma, lower, upper, algorithm='gibbs', position=None):
    """
    Sample from the truncated multivariate normal distribution.
         rpy2 - demo a Gibbs sampler in R when the package is available.
         scipy/emcee - demo a rejection sampler in Python when rpy2 is not available

    :param samples:        Number of samples to obtain (after burnin)
    :param mean:           Truncated normal mean vector
    :param sigma:          Truncated normal covariance matrix
    :param lower:          Lower bounds on of truncation
    :param upper:          Upper bounds of truncation
    :param algorithm:      Which rpy2+tmvtnorm algorithm to use.
                                (if not available use Affine Invariant MCMC Ensemble sampler, avoid in high dimensions)
    :param position:        Starting position for Markov Chain
    :return:
    """
    assert (sigma.shape[0] == sigma.shape[1] == len(mean))
    assert (len(lower) == len(upper) == len(mean))

    try:
        # Load the rpy2 package and load/install tmvtnorm lib
        return tmvtnorm_rpy2(samples, mean, np.ndarray.flatten(sigma), lower, upper, algorithm=algorithm, pos=position)
    except:
        #logging.info('rpy2 and/or tmvtnorm failed, resorting to rejection sampling')
        logging.critical('Not attempting to use emcee...')

        #try:
            # If the response is univariate we can use the scipy package for rejection sampling, not implemented
            # Otherwise we can try to use the emcee package for rejection sampling
        #    return tmvtnorm_emcee(samples, mean, sigma, lower, upper, pos=position)
        #except:
        raise ImportError('rpy2/emcee must be installed')


def tmvtnorm_rpy2(samples, mean, sigma_vec, lower, upper, algorithm='gibbs', pos=None):
    """
    Sampling the truncated multivariate normal distribution using Gibbs sampling.
    :return:
    """
    import rpy2.robjects.numpy2ri

    try:                                                                            # try and load library
        _ = importr_tryhard('mvtnorm')
        tmvtnorm = importr_tryhard('tmvtnorm')
    except:
        raise RuntimeError('Failed to import tmvtnorm and dependencies. Ensure R version > 3.')

    rpy2.robjects.numpy2ri.activate()                                               # activate pipe to demo R code
                                                                                    # convert args into R objects
    rmean = rpy2.robjects.FloatVector(mean)                                         # mean vector
    v = rpy2.robjects.FloatVector(sigma_vec)                                        # vectorised sigma
    rsigma = rpy2.robjects.r['matrix'](v, nrow=len(mean))                           # sigma matrix
    rlower = rpy2.robjects.FloatVector(lower)                                       # lower bound vector
    rupper = rpy2.robjects.FloatVector(upper)                                       # upper bound vector

    if pos is not None:
        rpos0 = rpy2.robjects.FloatVector(pos)                                      # Convert position into R object
        from rpy2.robjects.functions import SignatureTranslatedFunction             # Change arg signature for '.' args
        STM = SignatureTranslatedFunction                                           # TODO: check intricacies here
        tmvtnorm.rtmvnorm = STM(tmvtnorm.rtmvnorm,
                                init_prm_translate={'start_value': 'start.value'})

        return np.matrix(tmvtnorm.rtmvnorm(n=samples, mean=rmean, sigma=rsigma, lower=rlower, upper=rupper,
                                           algorithm=algorithm, start_value=rpos0))
    else:
        return np.matrix(tmvtnorm.rtmvnorm(n=samples, mean=rmean, sigma=rsigma, lower=rlower, upper=rupper,
                                           algorithm=algorithm))


def tmvtnorm_emcee(samples, mean, sigma, lower, upper, pos=None, burnin=10000):
    """
    Sampling the truncated multivariate normal distribution using rejection sampling with an ensemble sampler.
    see:  Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler.
    :return:
    """

    from numpy.linalg import inv
    import emcee

    if len(mean) > 10:
        logging.warning('Sampling in {0} dimensional space, install rpy2 for Gibb\'s sampling!'.format(len(mean)))
        logging.critical('Not attempting to use rejection sampling...')

    def lnprob_trunc_norm(x, mu, lower, upper, icov):
        if np.any(x < lower) or np.any(x > upper):
            return -np.inf
        else:
            diff = x - mu
            return -np.dot(diff, np.dot(icov, diff)) / 2.0

    # create ensemble sampler
    Nwalkers = 10 * len(mean)
    S = emcee.EnsembleSampler(Nwalkers, len(mean), lnprob_trunc_norm, a=20, args=(mean, lower, upper, inv(sigma)))

    # inital position for each walker, sample uniformly.
    #    If one bound is +/- inf then we create a small box around other bound.
    #    If both bounds are inf, then we set upper bin to mean and lower to mean - 0.1
    if pos is None:
        pos = np.ones((Nwalkers,len(mean)))
        for dim in range(len(mean)):
            low = lower[dim]
            upp = upper[dim]
            if lower[dim] == -np.inf:
                if upper[dim] == np.inf:
                    low = mean[dim]
                else:
                    low = upper[dim] - 1
            if upper[dim] == np.inf:
                upp = low + 1
            pos[:, dim] = np.random.uniform(low, upp, Nwalkers)
    else:
        assert len(pos) == len(mean)
        pos = np.tile(pos, (Nwalkers)).T

    walkerSamples = np.ceil(samples/Nwalkers)
    S.run_mcmc(pos, walkerSamples+burnin)

    # fill chain with values after burnin from each walker
    chain = np.zeros((samples, len(mean)))
    for walker in range(Nwalkers):
        bl = int(walker*walkerSamples)
        bu = int((walker+1)*walkerSamples)
        chain[bl:bu, :] = S.chain[walker, burnin:, :]

    # report acceptance rate
    if np.mean(S.acceptance_fraction) < 1:
        logger.warning(
            "Low mean acceptance rate: {0:.3f}. Too many dimensions?".format(np.mean(S.acceptance_fraction)))

        if len(mean) < 10 and False:
            import matplotlib.pyplot as plt
            for walker in range(S.chain.shape[0]):
                print(S.acceptance_fraction[walker])

                for latenty in range(S.chain.shape[2]):
                    if np.var(S.chain[walker, : ,latenty]) > 0:
                        print(np.var(S.chain[walker,:,latenty]))
                        plt.plot(range(S.chain.shape[1]), S.chain[walker, :, latenty])
                plt.show()

        if False:
            import matplotlib.pyplot as plt
            print(chain[0, :])
            plt.plot(range(chain.shape[1]), chain[0, :])
            plt.show()

    return chain[:samples, :]
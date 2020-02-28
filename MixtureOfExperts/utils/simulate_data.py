from __future__ import division
import numpy as np

__all__ = ['input_m1', 'mixture', 'santner', 'santner_mixture']


def input_m1(P, samples, m=4, var=4, corr1 = 3.5, corr2 = 3.5):
    """
    Input model one: generate samples independently and identically distributed as multivariate normal with mean and
    covariance as defined in Improving Prediction from Dirichlet Process Mixtures via Enrichment, equation 15.

    :param covariates:    number of covariate dimensions
    :param samples:       number of samples to generate
    :return:              simulated covariates
    """
    mean = m * np.ones(P)
    covariance = np.zeros((P, P))

    #group1 = [x for x in range(int(2*np.floor(P/2.))) if x%2 == 1 or x==0]
    #print(group1)
    #group2 = [x for x in range(int(2*np.floor((P-1)/2.) + 1)) if x%2 == 0 and x!=0]
    #print(group2)

    # use a loop for now, TODO: replace with more pythonic code
    for i in range(P):
        for j in range(P):
            if i == j:
                covariance[j, i] = var
            elif (i % 2 == 1 or i == 0) and (j % 2 == 1 or j == 0):
                covariance[j,i] = corr1
            elif (i % 2 == 0 and i != 0) and (j % 2 == 0 and j != 0):
                covariance[j,i] = corr2
            else:
                covariance[j,i] = 0

    return np.random.multivariate_normal(mean, covariance, samples)

def input_m1_cond(P, Xj, m=4, var=4, corr1 = 3.5, corr2 = 3.5):
    """
    Input model one: generate samples independently and identically distributed as multivariate normal with mean and
    covariance as defined in Improving Prediction from Dirichlet Process Mixtures via Enrichment, equation 15.

    These samples are conditional on an initial set

    :param covariates:    number of covariate dimensions
    :param X:             conditional samples
    :return:              simulated covariates
    """
    print(P)
    mean = m * np.ones(P)
    covariance = np.zeros((P, P))

    # group1 = [x for x in range(int(2*np.floor(P/2.))) if x%2 == 1 or x==0]
    # print(group1)
    # group2 = [x for x in range(int(2*np.floor((P-1)/2.) + 1)) if x%2 == 0 and x!=0]
    # print(group2)

    # use a loop for now, TODO: replace with more pythonic code
    for i in range(P):
        for j in range(P):
            if i == j:
                covariance[j, i] = var
            elif (i % 2 == 1 or i == 0) and (j % 2 == 1 or j == 0):
                covariance[j, i] = corr1
            elif (i % 2 == 0 and i != 0) and (j % 2 == 0 and j != 0):
                covariance[j, i] = corr2
            else:
                covariance[j, i] = 0
    #print(covariance)

    muj = mean[0:Xj.shape[1]]
    mui = mean[Xj.shape[1]:]
    sigmajj = covariance[0:Xj.shape[1], 0:Xj.shape[1]]
    sigmaii = covariance[Xj.shape[1]-1:-1, Xj.shape[1]-1:-1]
    sigmaji = covariance[0:Xj.shape[1], Xj.shape[1]:]
    sigmaij = np.transpose(sigmaji)

    #print(np.hstack((muj, mui)))
    #print(np.vstack((np.hstack((sigmajj, sigmaji)), np.hstack((sigmaij, sigmaii)))))

    # Mean and variance of covariates given
    cond_mean = np.zeros((Xj.shape[0], mui.shape[0]))
    for sample in range(Xj.shape[0]):
        cond_mean[sample :] = mui + np.dot(sigmaij, np.dot(np.linalg.inv(sigmajj),(Xj[sample, :] - muj)))
    cond_sig = sigmaii - np.dot(np.transpose(sigmaji), np.dot(np.linalg.inv(sigmajj), sigmaji))

    Xi = np.zeros((Xj.shape[0], mui.shape[0]))
    for sample in range(Xj.shape[0]):
        Xi[sample, :] = np.random.multivariate_normal(cond_mean[sample, :], cond_sig, 1)

    X = np.hstack((Xj, Xi))

    #print(X.shape)
    #print(np.mean(X))
    #print(np.cov(X, rowvar=False))
    #print(covariance)

    return X

def mixture(X, mu1=0.25, mu2=0.75, tau1=5., tau2=5., plot = None):
    """
    Calculate the mixture probabilities for a two class mixture model as a function of the first covariate. The mixture
    probability depends only on the first covariate of input data. See Improving Prediction from Dirichlet Process
    Mixtures via Enrichment, simulated example.

    We do not use a package's density function to make parameterisation comparisons to the paper above simple.

    :param X:       input data ( samples * covariates )
    :param mu1:
    :param mu2:
    :param tau1:
    :param tau2:
    :param plot:    plotting, three options: None, do not plot; True, plot and show; string, plot and save to string
    :return:         vector of ratio probabilities p(x_{:,1}) where each element is for a different input sample.
    """
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    x1 = X[:,0]

    gauss = lambda x, tau, mu: tau * np.exp(-(tau**2/2) * (x-mu)**2)

    g1 = gauss(x1, tau1, mu1)
    g2 = gauss(x1, tau2, mu2)

    if plot is not None:
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError('cannot import matplotlib')

        plt.scatter(x1, g1 / (g1+g2))
        plt.scatter(x1, 1-(g1 / (g1+g2)))
        if plot is True:
            plt.show()
        else:
            plt.savefig(plot+'_mixture')
        plt.close()

    return g1 / (g1+g2)

def santner(X, noise=0):
    """
    Simulate data from the damped cosine function from Santner et al.

    :param X:
    :param noise:
    :return:
    """
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))

    Y = np.exp(-1.4 * X[:, 0]) * np.cos(((X[:, 0] > 0.4) * 3.5 + (X[:, 0] <= 0.4) * 10.0) * (np.pi * X[:, 0])) + \
        noise * np.random.randn(len(X[:, 0]))

    return np.reshape(Y, (-1,1))

def santner_mixture(X, weights=None, factor1=3.5, factor2=10., coef1=-1.4, coef2=-1.4, noise=0.05, plot=None):
    """
    Simulate data from the damped cosine function from Santner et al. However, instead of using a step function to
    choose the cosine function, we sample the allocating probability using a mixture model giving assignment
    probabilities as a function of the input.

    :param X:          input data
    :param factor1:    scalars for factor in first and second component. Also proportional to first covariate.
    :param factor2:
    :param coef1:      scalar coefficient for first and second component. Also proportional to first covariate.
    :param coef2
    :param mix_han:    handle to a 2 class mixture model which returns assignment probabilities for the first class
    :param noise:
    :return:
    """

    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    if weights is None:
        weights = mixture(X)

    uniform = np.random.uniform(0, 1, X.shape[0])
    assignment = (weights < uniform)

    Y = assignment * np.exp(coef1 * X[:, 0]) * np.cos(factor1 * np.pi * X[:, 0])
    Y += (1-assignment) * np.exp(coef2 * X[:, 0]) * np.cos(factor2 * np.pi * X[:, 0])

    if np.isscalar(noise):
        Y += noise * np.random.randn(len(X[:, 0]))
    elif len(noise) == 2:
        Y += assignment * noise[0] * np.random.randn(len(X[:, 0]))
        Y += (1-assignment) * noise[1] * np.random.randn(len(X[:, 0]))

    if plot is not None:
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError('cannot import matplotlib')

        plt.scatter(X[assignment, 0], Y[assignment])
        plt.scatter(X[np.invert(assignment), 0], Y[np.invert(assignment)])
        if plot is True:
            plt.show()
        else:
            plt.savefig(plot+'_santner_mixture')
        plt.close()

    return np.reshape(Y, (-1,1)), assignment

def binary_mixture(X, weights=None, plot=None):
    """ Simulate data from {0,1} with allocation probability given by a mixture model giving assignment
    probabilities as a function of the input.

    :param X:          input data
    """

    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    if weights is None:
        weights = mixture(X)

    uniform = np.random.uniform(0, 1, X.shape[0])
    assignment = (weights < uniform)
    Y = assignment*1

    if plot is not None:
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError('cannot import matplotlib')

        plt.scatter(X[assignment, 0], Y[assignment])
        plt.scatter(X[np.invert(assignment), 0], Y[np.invert(assignment)])
        if plot is True:
            plt.show()
        else:
            plt.savefig(plot+'_binary_mixture')
        plt.close()

    return np.reshape(Y, (-1,1)), assignment

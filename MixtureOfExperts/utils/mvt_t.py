from __future__ import division
import math
import numpy as np
import logging
logger = logging.getLogger(__name__)

__all__ = ['multivariate_t_distribution_pdf']


def multivariate_t_distribution_pdf(x, mu, Sigma, df):
    """
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        d: dimension
    """
    x = np.asarray(x.reshape((-1, 1)))
    mu = np.asarray(mu.reshape((-1, 1)))

    assert (mu.shape == x.shape) and (mu.shape[0] == Sigma.shape[0])
    assert Sigma.shape[0] == Sigma.shape[1]
    assert Sigma.shape[0] > 1

    p = Sigma.shape[0]

    density = math.gamma(1. * (p + df) / 2)
    density /= (math.gamma(1. * df / 2) * pow(df * math.pi, 1. * p / 2) * pow(np.linalg.det(Sigma), 1. / 2))
    density /= pow(1 + (1. / df) * np.dot(np.dot((x - mu).T, np.linalg.inv(Sigma)), (x - mu)), (p + df) / 2)

    return density

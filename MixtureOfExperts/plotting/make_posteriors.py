import MixtureOfExperts
import numpy as np
import matplotlib.pyplot as plt
import os, sys


def load_enriched(name):
    raise NotImplementedError('Deprecated')
    model = MixtureOfExperts.models.EnrichedMixtureOfExperts(np.ones((1, 1)), np.ones((1, 1)), None, None, name=name)
    # shouldn't reach this as the way we load will raise an exception if it can't load. TODO: add better loading
    assert model._mcmcSteps > 0, 'Was unable to load, ensure you are in the same directory.'

    return model


def load_joint(name):
    raise NotImplementedError('Deprecated')
    model = MixtureOfExperts.models.JointMixtureOfExperts(np.ones((1, 1)), np.ones((1, 1)), None, None, name=name)
    # shouldn't reach this as the way we load will raise an exception if it can't load. TODO: add better loading
    assert model._mcmcSteps > 0, 'Was unable to load, ensure you are in the same directory.'

    return model


def plot_posteriors(model, start=0, stop=None, indexList=None, bins=500, save=False):
    """ Make prior/posterior plots out of a Markov Chain """
    print('Making posterior plots for ' + model.__name__)

    if stop == None:
        stop = model._mcmcSteps - 1

    names = model._expert.param_names

    states = model.states
    thetaPosterior = np.empty((0, model._expert.n_hyper), int)
    for state in np.linspace(start, stop, stop-start+1):
        clustersJ = np.asarray(np.unique(states[int(state)]['sIndex'][:, 0]), dtype=int)
        thetaJ = states[int(state)]['theta'][clustersJ, :]
        thetaPosterior = np.vstack([thetaPosterior, thetaJ])

    thetaPrior = model._expert.prior(thetaPosterior.shape[0])

    if indexList is None:
        ploti = range(model._expert.n_hyper)
    else:
        ploti = indexList[0]

    for i in ploti:
        plt.subplot(121)
        plt.hist(thetaPrior[:, i], bins=bins, normed=True)
        plt.title(names[i] + ' prior')
        plt.subplot(122)
        plt.hist(thetaPosterior[:, i], bins=bins, normed=True)
        plt.title(names[i] + ' posterior')
        if save is False:
            plt.show()
        else:
            plt.savefig(save + 'hyper{0}'.format(i))
            plt.close()

    return 1

if __name__ == '__main__':
    # must demo from directory model is saved in.


    name = 'EmE1_init6_6'       # 5=1, 2=4.5, 1=5
    model = load_enriched(name)
    plot_posteriors(model)
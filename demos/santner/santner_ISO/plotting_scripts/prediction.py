import numpy as np

def predict_density(model, Xtest, feature=0, data_gen_fun=None, mean=False, density=False, burnin=0, MC=None, save=None):
    """
    Predict the density and compute error for a DP or EDP model for the Isotropic example, where the data generating
    function is a function of the mean of the input samples.

    :param model:                           EDP or DP model
    :param Xtest:                           The matrix of test covariates
    :param feature:                         The feature to plot, (one dimensional example so default 0)
    :param data_gen_fun:                    The true data generating function
    :param mean:                            Boolean for if we should plot the model predictive mean
    :param density:                         Boolean for if we should plot the model predictive density
    :param burnin:                          The burn-in used from the MCMC chain (that must already have been ran)
    :param MC:                              How far along the chain to go, if None defaults to the full chain
    :param save:                            Save or plot the results
    :return:
    """
    try:
        import matplotlib.pyplot as plt
    except:
        raise ImportError('cannot import matplotlib')

    boundl = np.min(model.y[:, feature])
    boundu = np.max(model.y[:, feature])

    numStates = len(model.states)
    assert numStates > 0, 'plot_predict: Need to iterate over points at least once.'  # otherwise theta == 0
    assert model.ydim > feature, 'plot_predict: Cannot plot feature {0} as we have {1} features'.format(
        feature, model.ydim)

    # data generating function
    if data_gen_fun is not None:
        plt.plot(np.mean(Xtest, axis=1), data_gen_fun(Xtest, 0), 'k--')

    # mean predictions of the state
    if mean is True:
        M = model._predict_expectation(Xtest, burnin=burnin, MC=MC)
        plt.plot(np.mean(Xtest, axis=1), M[feature], 'b')

    if density is True:
        [Xgrid, Ygrid, D] = model._predict_density(Xtest, burnin=burnin, MC=MC)
        Dgrid = D[feature]
        plt.pcolormesh(np.asarray(Xgrid), Ygrid, Dgrid, cmap='Reds')
        plt.colorbar()
        boundl = np.min(Ygrid[:, 0])
        boundu = np.max(Ygrid[:, 0])

    # scatter plot data points
    plt.scatter([np.mean(model.x, axis=1)], [model.y[:, feature]], c='k')

    plt.axis([np.min(model.x), np.max(model.x), boundl, boundu])
    plt.title('Monte Carlo predictive density')
    plt.xlabel('x mean')
    plt.ylabel('y_{0}'.format(feature))

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()

    plt.close()
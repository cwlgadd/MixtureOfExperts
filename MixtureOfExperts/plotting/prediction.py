"""
This file contains functions for plotting various figures summarising the models included within this package.

The functions contain sensible defaults for methods shared across each model.
"""

#TODO: this needs to be checked over after some heavy changes to the functions it calls.
#TODO: state_summary: densities are not found for only the state given, change default to fix this

from __future__ import division
import numpy as np

def state_summary(model, state=-1, data_gen_fun=None, GPs=False, mean=False, density=False, covariate=0, feature=0,
                  save=None):
    """
    A summary plot for a single state of the Markov Chain.

    This plot has options to show:
            to show the data generating function
            the predive moments of the GPs for each cluster
            the mean prediction of the weighted GPs.
            the density prediction of the weighted GPs.
    This always plots the data (relevant splices) and labels the data points with the cluster allocations. For
    predictions, we vary along one covariate, with others fixed at the mean value, and give predictions for one feature.

    TODO: allow vector feature as this won't change computational cost.

    :param state:               index of state we wish to observe (defaults to -1, last state in the chain)
        :type                   int
    :param data_gen_fun:        data generating function
        :type                   function handle
    :param GPs:                 whether we plot the cluster GPs predictive mean/var
        :type                   boolean
    :param mean:                whether we plot the model mean (weighted GPs) for this cluster.
        :type                   boolean
    :param density              whether we plot the predictive density for this state.
        :type                   boolean
    :param covariate:           the index of the covariate we want to make predictions for
        :type                   int
    :param feature:             the index of the feature we want to view predictions of
        :type                   int
    """

    try:
        import matplotlib.pyplot as plt
    except:
        raise ImportError('cannot import matplotlib')

    # extract the information from the list of dictionaries model._MCMCstates.
    kCopy = model.states[state]['k_counter']     # number of clusters
    sIndexCopy = model.states[state]['sIndex']  # allocation of data points to model._k_counter clusters
    thetaCopy = model.states[state]['theta']

    # allow for both one and two layer models.
    if sIndexCopy.ndim == 1:
        sIndexCopy = np.reshape(sIndexCopy, (-1,1))

    # create figure
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.axis([np.min(model.x[:, covariate]), np.max(model.x[:, covariate]),# -10,30])
             np.min(model.y[:,feature])-1.5, np.max(model.y[:,feature])+1.5])
    # get colour palette
    itertool = ax._get_lines.prop_cycler

    # allocate test points for optional plots at end
    test_points = 200
    Xtest = np.matlib.repmat(np.reshape(np.mean(model.x, axis=0), (1, -1)), test_points, 1)
    Xtest[:, covariate] = np.linspace(np.min(model.x[:, covariate]), np.max(model.x[:, covariate]), test_points)

    # plot data generating function
    if data_gen_fun is not None:
        ax.plot(Xtest[:, covariate], data_gen_fun(Xtest, 0)[:, feature], 'k')

    # plots which are cluster specific
    for k in range(kCopy):  # for each cluster
        if np.sum(sIndexCopy[:, 0] == k) > 0:  # if not empty
            xk = model.x[sIndexCopy[:, 0] == k, :]
            ykf = model.y[sIndexCopy[:, 0] == k, feature]
            color = next(itertool)['color']  # update colour palette

            # scatter plot data points with their cluster allocations annotated
            ax.scatter([xk[:, covariate]], [ykf], s=50, c=color,
                       label='cluster {0}: var{1:.3f}, len{2:.3f}, nvar{3:.3f}'.format(k, thetaCopy[k, 0],
                                                                                       thetaCopy[k, 1],
                                                                                       thetaCopy[k, 2]))
            for i, txt in enumerate(xk):
                ax.annotate(str(k), (xk[i, covariate], ykf[i] + 0.01))

            # plot individual GP mean and error bar for each non-empty cluster
            if GPs is True:
                #assert len(model.states) > 1, 'plot_state_summary: Need to update posterior at least once.'

                mu, std = model._expert.predict(xk, np.reshape(ykf, (-1, 1)), Xtest, thetaCopy[k, :])

                ax.plot(Xtest[:, covariate], mu[:, feature], color)
                ax.fill_between(Xtest[:, covariate], (mu[:, feature] - 2*std[:, feature]).flatten(),
                                (mu[:, feature] + 2*std[:, feature]).flatten(), color=color,
                                alpha=0.2)

                #model._expert.plot_1d(xk, np.reshape(ykf, (-1, 1)), thetaCopy[k, :], 0)

    # plot mean prediction of the state
    if mean is True:
        #assert len(model.states) > 1, 'plot_state_summary: Need to update posterior at least once.'
        mean = model._predict_expectation(Xtest)
        ax.plot(Xtest[:, covariate], mean[feature], 'k--')

    # plot predictive density of the state
    if density is True:
        [Xgrid, Ygrid, D] = model._predict_density(Xtest, covariate=covariate)
        ax.axis([np.min(model.x[:, covariate]), np.max(model.x[:, covariate]), np.min(Ygrid[:, 0]), np.max(Ygrid[:, 0])])
        plt.pcolormesh(Xgrid, Ygrid, D[feature], cmap='Reds')
        plt.colorbar()

    # ax.legend()#loc='center left', bbox_to_anchor=(0.5, 0.5))
    plt.title('Mean predictions for state {0}'.format(state))

    if save is not None:
        plt.savefig(save)
        plt.close()
    #else:
    #    plt.show()
    #    plt.close()

    #plt.close()

    return ax


def predict(model, covariate=0, feature=0, data_gen_fun=None, mean=False, density=False, burnin=0, MC=None, save=None):
    """
    Plot the predictions of our MC. We vary along one covariate, with others fixed at the mean value, and give
    predictions for one feature.

    This plot has options to show:
            the data generating function
            the mean function
            the feature density.
    This always plots the data (relevant splices).

    TODO: allow vector feature as this won't change computational cost.
    TODO: move this to the plotting module, so the function does not need to be repeatedly defined in each model.

    :param covariate:           the index of the covariate we want to make predictions for
        :type                   int
    :param feature:             the index of the feature we want to view predictions of
        :type                   int
    :param data_gen_fun:        data generating function
        :type                   function handle
    :param mean:                whether we plot the model mean (weighted GPs) for this cluster.
        :type                   boolean
    :param density              whether we plot the predictive density for this state.
        :type                   boolean
    """
    try:
        import matplotlib.pyplot as plt
    except:
        raise ImportError('cannot import matplotlib')

    boundl = np.min(model.y[:, feature])
    boundu = np.max(model.y[:, feature])

    numStates = len(model.states)
    assert numStates > 0, 'plot_predict: Need to iterate over points at least once.' # otherwise theta == 0
    assert model.xdim > covariate, 'plot_predict: Cannot plot covariate {0} as we have {1} covariates'.format(
                                    covariate, model.xdim)
    assert model.ydim > feature, 'plot_predict: Cannot plot feature {0} as we have {1} features'.format(
                                    feature, model.ydim)

    # allocate test points for optional plots below
        # we vary the tests input dimension of interest, whilst keeping others fixed at their mean values.
    Xtest = np.matlib.repmat(np.mean(model.x, axis=0), 200, 1)
    Xtest[:, covariate] = np.linspace(np.min(model.x), np.max(model.x), 200)

    # data generating function
    if data_gen_fun is not None:
        plt.plot(Xtest[:, covariate], data_gen_fun(Xtest, 0), 'k--')

    # mean predictions of the state
    if mean is True:
        M = model._predict_expectation(Xtest, burnin=burnin, MC=MC)
        plt.plot(Xtest[:,covariate], M[feature], 'b')

    if density is True:
        [Xgrid, Ygrid, D] = model._predict_density(Xtest, covariate=covariate, burnin=burnin, MC=MC)
        Dgrid = D[feature]
        plt.pcolormesh(np.asarray(Xgrid), Ygrid, Dgrid, cmap='Reds')
        plt.colorbar()
        boundl = np.min(Ygrid[:,0])
        boundu = np.max(Ygrid[:,0])

    # scatter plot data points
    plt.scatter([model.x[:, covariate]], [model.y[:, feature]], c='k')

    plt.axis([np.min(model.x), np.max(model.x), boundl, boundu])
    plt.title('Monte Carlo predictive density')
    plt.xlabel('x_{0}'.format(covariate))
    plt.ylabel('y_{0}'.format(feature))

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()

    plt.close()


def predict_marginal(model, covariate=0, feature=0, data_gen_fun=None, mean=False, density=False, burnin=0, MC=None, save=None):
    """
    Plot the predictions of our MC. We vary along one covariate, with others fixed at the mean value, and give
    predictions for one feature.

    This plot has options to show:
            the data generating function
            the mean function
            the feature density.
    This always plots the data (relevant splices).

    TODO: allow vector feature as this won't change computational cost.
    TODO: move this to the plotting module, so the function does not need to be repeatedly defined in each model.

    :param covariate:           the index of the covariate we want to make predictions for
        :type                   int
    :param feature:             the index of the feature we want to view predictions of
        :type                   int
    :param data_gen_fun:        data generating function
        :type                   function handle
    :param mean:                whether we plot the model mean (weighted GPs) for this cluster.
        :type                   boolean
    :param density              whether we plot the predictive density for this state.
        :type                   boolean
    """
    try:
        import matplotlib.pyplot as plt
    except:
        raise ImportError('cannot import matplotlib')

    boundl = np.min(model.y[:, feature])
    boundu = np.max(model.y[:, feature])

    numStates = len(model.states)
    assert numStates > 0, 'plot_predict: Need to iterate over points at least once.' # otherwise theta == 0
    assert model.xdim > covariate, 'plot_predict: Cannot plot covariate {0} as we have {1} covariates'.format(
                                    covariate, model.xdim)
    assert model.ydim > feature, 'plot_predict: Cannot plot feature {0} as we have {1} features'.format(
                                    feature, model.ydim)

    # allocate test points for optional plots below
        # we vary the tests input dimension of interest, whilst keeping others fixed at their mean values.
    Xtest = np.reshape(np.linspace(np.min(model.x), np.max(model.x), 200), (-1,1))

    # mean predictions of the state
    if mean is True:
        M = model._predict_marginal_expectation(Xtest, covariate, burnin=burnin, MC=MC)
        print(M[feature])
        plt.plot(Xtest, M[feature], 'b')

    if density is True:
        [Xgrid, Ygrid, D] = model._predict_marginal_density(Xtest, covariate, burnin=burnin, MC=MC)
        Dgrid = D[feature]
        plt.pcolormesh(np.asarray(Xgrid), Ygrid, Dgrid, cmap='Reds')
        plt.colorbar()
        boundl = np.min(Ygrid[:,0])
        boundu = np.max(Ygrid[:,0])

    # scatter plot data points
    plt.scatter([model.x[:, covariate]], [model.y[:, feature]], c='k')

    plt.axis([np.min(model.x), np.max(model.x), boundl, boundu])
    plt.title('Monte Carlo predictive density')
    plt.xlabel('x_{0}'.format(covariate))
    plt.ylabel('y_{0}'.format(feature))

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()

    plt.close()


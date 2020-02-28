base = "/l/gaddc1/Dropbox/"
root = "MixtureOfExperts/demos/santner/santner_ARD/ARD/Apr15_fromTriton/"

import sys, os
import numpy as np
import scipy.stats as stats
import MixtureOfExperts
from MixtureOfExperts.utils import simulate_data as sd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import pickle

def plots_cond(model, xbins, ybins, burnin=500, MC=None, returnE=False, thinning=1):

    Xtest = np.zeros((len(xbins), model.xdim))
    for pplus in range(model.xdim-1):
        Xtest[:, pplus + 1] = np.mean(model.x, axis=0)[pplus + 1]
    Xtest[:, 0] = xbins

    dDict = model._predict_density(Xtest, y_test=ybins, covariate=0, burnin=burnin, MC=MC, thinning=thinning)
    dDict['D_cond'] = dDict.pop('D')
    if returnE is True:
        eDict = model._predict_expectation(Xtest, burnin=burnin, MC=MC, thinning=thinning)
        eDict['E_cond'] = eDict.pop('E')
        dDict = {**dDict, **eDict}

    return dDict

def plots_sample(model, Xsamp, ybins, burnin=500, MC=None, returnE=False, thinning=1):

    dDict = model._predict_density(Xsamp, y_test=ybins, covariate=0, burnin=burnin, MC=MC, thinning=thinning)
    dDict['D_samp'] = dDict.pop('D')
    dDict = {**dDict, 'Xsamp':Xsamp}
    if returnE is True:
        eDict = model._predict_expectation(Xsamp, burnin=burnin, MC=MC, thinning=thinning)
        eDict['E_samp'] = eDict.pop('E')
        dDict = {**dDict, **eDict}

    Xgrid_samp, Ygrid_samp = np.meshgrid(Xsamp[:, 0], ybins)
    dDict = {**dDict, 'Xgrid_samp':Xgrid_samp, 'Ygrid_samp':Ygrid_samp}

    return dDict

def plots_marg(model, xbins, ybins, burnin=500, MC=None, returnE=False, thinning=1):

    dDict = model._predict_marginal_density(np.reshape(xbins,(-1,1)), y_test=ybins, covariate=0, burnin=burnin,
                                            MC=MC, thinning=thinning)
    dDict['D_marg'] = dDict.pop('D')

    if returnE is True:
        eDict = model._predict_marginal_expectation(np.reshape(xbins,(-1,1)), covariate=0, burnin=burnin, MC=MC,
                                                    thinning=thinning)
        eDict['E_marg'] = eDict.pop('E')
        dDict = {**dDict, **eDict}

    return dDict

def data_generating_function_density(xbins, ybins, mu1=3., mu2=5., tau1=0.8, tau2=0.8, coef1=0.1, coef2=-0.1,
                                     factor1=0.6, factor2=0.4):
    """
    Get the density (and plot) the true data generating function. Return density and
    """

    Xgrid, Ygrid = np.meshgrid(xbins, ybins)  # Grid for density (#y * #x)
    density_true = np.zeros_like(Xgrid)

    weights = sd.mixture(xbins, mu1=mu1, mu2=mu2, tau1=tau1, tau2=tau2)

    for xi in range(xbins.shape[0]):
        mean1 = np.exp(coef1 * xbins[xi]) * np.cos(factor1 * np.pi * xbins[xi])
        mean2 = np.exp(coef2 * xbins[xi]) * np.cos(factor2 * np.pi * xbins[xi])
        density_true[:, xi] += weights[xi] * stats.norm.pdf(ybins, loc=mean2, scale= 0.05)
        density_true[:, xi] += (1-weights[xi]) * stats.norm.pdf(ybins, loc=mean1, scale= 0.15)

    return density_true

def errors(model, Xtest, burnin=500, MC=None, plot=False, thinning=1, returnE=False, loadDict=False):
    """
    """

    xbins = np.linspace(np.min(model.x), np.max(model.x), 200)
    ybins = np.linspace(np.min(model.y)*1.2, np.max(model.y)*1.2, 500)
    Xgrid, Ygrid = np.meshgrid(xbins, ybins)

    # Make/get conditional predictions for expectation/density over (latent) response
    if loadDict is True:
        try:
            with open('density_dictionary.pkl', 'rb') as f:
                errorDict = pickle.load(f)
        except:
            raise RuntimeError('Unable to load marginal prediction dictionary')
    else:

        margDict = plots_marg(model, xbins, ybins, burnin=burnin, MC=MC, returnE=returnE, thinning=thinning)
        condDict = plots_cond(model, xbins, ybins, burnin=burnin, MC=MC, returnE=returnE, thinning=thinning)
        sampDict = plots_sample(model, Xtest, ybins, burnin=burnin, MC=MC, returnE=returnE, thinning=thinning)

        density_true = data_generating_function_density(xbins, ybins)
        density_samp_true = data_generating_function_density(sampDict['Xsamp'][:, 0], ybins)
        errorDict = {**condDict, **margDict, **sampDict, 'density_true':density_true,
                     'density_samp_true':density_samp_true}

        with open('density_dictionary.pkl', 'wb') as f:
            pickle.dump(errorDict, f, pickle.HIGHEST_PROTOCOL)

    # Unpack
    density_cond = errorDict['D_cond'][0]
    density_marg = errorDict['D_marg'][0]
    density_samp = errorDict['D_samp'][0]
    Xgrid_samp = errorDict['Xgrid_samp']
    Ygrid_samp = errorDict['Ygrid_samp']
    density_true = errorDict['density_true']
    density_samp_true = errorDict['density_samp_true']
    if {'E_cond', 'E_marg', 'E_samp'} <= errorDict.keys():
        expectation_cond = errorDict['E_cond'][0]
        expectation_marg = errorDict['E_marg'][0]
        expectation_samp = errorDict['E_samp'][0]
        Xsamp = errorDict['Xsamp'][:, 0]

    # error_cond = np.linalg.norm(density_cond - density_true) / np.linalg.norm(density_true)
    error_cond = np.sum(np.abs(density_cond - density_true)) / np.prod(np.shape(density_true))
    # error_marg = np.linalg.norm(density_marg - density_true) / np.linalg.norm(density_true)
    error_marg = np.sum(np.abs(density_marg - density_true)) / np.prod(np.shape(density_true))
    # error_samp = np.linalg.norm(density_samp - density_samp_true) / np.linalg.norm(density_samp_true)
    error_samp = np.sum(np.abs(density_samp - density_samp_true)) / np.prod(np.shape(density_samp_true))

    if plot is not False:

        DensityYSlices = False
        Density = False
        Paper = True

        if DensityYSlices:
            plt.close('all')
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
            annotations = np.zeros((9, 10))

            for xi in range(10):
                area_true = np.trapz(np.reshape(density_true[:, 20 * xi], (1, -1)), dx=ybins[1] - ybins[0])
                area_cond = np.trapz(np.reshape(density_cond[:, 20 * xi], (1, -1)), dx=ybins[1] - ybins[0])
                area_marg = np.trapz(np.reshape(density_marg[:, 20 * xi], (1, -1)), dx=ybins[1] - ybins[0])

                ax1.plot(ybins, density_true[:, 20 * xi], label=area_true)
                ind = np.argmax(density_true[:, 20 * xi])
                annotations[0:3, xi] = [ybins[ind], density_true[ind, 20 * xi], area_true]

                ax2.plot(ybins, density_cond[:, 20 * xi], label=area_cond)
                ind = np.argmax(density_cond[:, 20 * xi])
                annotations[3:6, xi] = [ybins[ind], density_cond[ind, 20 * xi], area_cond]

                ax3.plot(ybins, density_marg[:, 20 * xi], label=area_marg)
                ind = np.argmax(density_marg[:, 20 * xi])
                annotations[6:, xi] = [ybins[ind], density_marg[ind, 20 * xi], area_marg]

            [ax1.annotate('{0:.1f}:{1:.2f}'.format(xbins[20 * xi], annotations[2, xi]), xy=annotations[0:2, xi], fontsize=7) for xi in range(10)]
            [ax2.annotate('{0:.1f}:{1:.2f}'.format(xbins[20 * xi], annotations[5, xi]), xy=annotations[3:5, xi], fontsize=8) for xi in range(10)]
            [ax3.annotate('{0:.1f}:{1:.2f}'.format(xbins[20 * xi], annotations[8, xi]), xy=annotations[6:8, xi], fontsize=8) for xi in range(10)]

            ax1.set_title('True')
            ax2.set_title('Conditional')
            ax3.set_title('Marginalised')

            if plot is True:
                plt.show()
            elif plot is 'save':
                plt.savefig('density_slices.png')
            else:
                raise ValueError
            plt.close('all')

        if Density:
            plt.close('all')

            plt.subplot(421)
            plt.pcolormesh(np.asarray(Xgrid), Ygrid, (density_true), cmap='Reds')
            plt.colorbar()
            plt.scatter(model.x[:, 0], model.y, s=0.1)
            plt.title('Truth')

            plt.subplot(422)
            plt.pcolormesh(np.asarray(Xgrid_samp), Ygrid_samp, (density_samp_true), cmap='Reds')
            plt.colorbar()
            #plt.scatter(Xsamp[:, 0], np.zeros_like(Xsamp[:, 0]), s=0.1)
            plt.title('Truth - sampling')

            plt.subplot(423)
            plt.pcolormesh(np.asarray(Xgrid), Ygrid, (density_cond), cmap='Reds')
            plt.colorbar()
            plt.scatter(model.x[:, 0], model.y, s=0.1)
            if 'expectation_cond' in locals():
                plt.plot(xbins, expectation_cond)
            plt.title('Conditional')

            plt.subplot(424)
            plt.pcolormesh(np.asarray(Xgrid), Ygrid, (np.abs(density_true - density_cond)), cmap='Reds')
            plt.colorbar()
            plt.scatter(model.x[:, 0], model.y, s=0.1)
            plt.title('|error|={0} conditional'.format(error_cond))

            plt.subplot(425)
            plt.pcolormesh(np.asarray(Xgrid), Ygrid, (density_marg), cmap='Reds')
            plt.colorbar()
            plt.scatter(model.x[:, 0], model.y, s=0.1)
            if 'expectation_marg' in locals():
                plt.plot(xbins, expectation_marg)
            plt.title('Model marginalised')

            plt.subplot(426)
            plt.pcolormesh(np.asarray(Xgrid), Ygrid, (np.abs(density_true - density_marg)), cmap='Reds')
            plt.colorbar()
            plt.scatter(model.x[:, 0], model.y, s=0.1)
            plt.title('|error|={0} marginalised'.format(error_marg))

            plt.subplot(427)
            plt.pcolormesh(np.asarray(Xgrid_samp), Ygrid_samp, (density_samp), cmap='Reds')
            plt.colorbar()
            if 'expectation_samp' in locals():
                plt.scatter(Xsamp, expectation_samp, s=0.1)
            plt.title('Model sampling')

            plt.subplot(428)
            plt.pcolormesh(np.asarray(Xgrid_samp), Ygrid_samp, (np.abs(density_samp_true - density_samp)), cmap='Reds')
            plt.colorbar()
            plt.title('|error|={0} sampling'.format(error_samp))

            if plot is True:
                plt.show()
            elif plot is 'save':
                plt.savefig('density_error.png')
            else:
                raise ValueError
            plt.close('all')

        if Paper:
            s = 0.75
            c = 'k'
            plt.close('all')

            fig, (ax1) = plt.subplots(ncols=1)

            cax = ax1.pcolormesh(np.asarray(Xgrid), Ygrid, density_true, cmap='Reds')
            fig.colorbar(cax, ax=ax1)
            ax1.scatter(model.x[:, 0], model.y, s=s, c=c)
            plt.xlabel('$x_{1}$')
            plt.ylabel('$y$')
            plt.xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
            plt.xlim((min(xbins), max(xbins)))
            plt.ylim(min(ybins), max(ybins))
            plt.tight_layout()
            if plot is True:
                plt.show()
            elif plot is 'save':
                plt.savefig('density_truth.eps', format='eps')
            else:
                raise ValueError
            plt.close('all')

            plt.figure()
            plt.pcolormesh(np.asarray(Xgrid), Ygrid, density_cond, cmap='Reds')
            plt.colorbar()
            plt.scatter(model.x[:, 0], model.y, s=s, c=c)
            if 'expectation_cond' in locals():
                plt.plot(xbins, expectation_cond, c=c, linewidth=0.75)
            plt.xlabel('$x_{1}$')
            plt.ylabel('$y$')
            plt.xlim((min(xbins), max(xbins)))
            plt.ylim(min(ybins), max(ybins))
            plt.tight_layout()
            #plt.xticks(fontsize=fontsize)
            #plt.yticks(fontsize=fontsize)
            if plot is True:
                plt.show()
            elif plot is 'save':
                plt.savefig('density_conditional.eps', format='eps')
            else:
                raise ValueError
            plt.close('all')

            plt.figure()
            plt.pcolormesh(np.asarray(Xgrid), Ygrid, density_marg, cmap='Reds')
            plt.colorbar()
            plt.scatter(model.x[:, 0], model.y, s=s, c=c)
            if 'expectation_marg' in locals():
                plt.plot(xbins, expectation_marg, c=c, linewidth=0.75)
            plt.xlabel('$x_{1}$')
            plt.ylabel('$y$')
            plt.xlim((min(xbins), max(xbins)))
            plt.ylim(min(ybins), max(ybins))
            plt.tight_layout()
            #plt.xticks(fontsize=fontsize)
            #plt.yticks(fontsize=fontsize)
            if plot is True:
                plt.show()
            elif plot is 'save':
                plt.savefig('density_marginal.eps', format='eps')
            else:
                raise ValueError
            plt.close('all')

    return error_cond, error_marg, error_samp


if __name__ == '__main__':
    """
    Generate the paper plots and metrics relevant to the predictive error
    """

    # Pre-sample test points for fair comparison
    test_data = np.loadtxt(base + root + f"santner_mixture_test_new.csv", delimiter=", ")
    x_test = test_data[:, 0:-2]

    for p in [1, 2, 5, 10, 15, 20]:
        print(f'\nRunning error plots script for p={p}')
        if p not in []:
            load = True
        else:
            load = True

        # Create coverage plots
        os.chdir(base + root + f'JmE/JmE{p}_initS_linexam')
        modelJmE = MixtureOfExperts.models.load_model('JmE', f'JmE{p}_initS_linexam', root=os.getcwd()+'/')
        jc1, jm1, js1 = errors(modelJmE, x_test[:, 0:p], burnin=1000, MC=np.min((5000, modelJmE._mcmcSteps)),
                               plot='save', thinning=10, returnE=True, loadDict=load)
        print(f'{jc1}, {jm1}, {js1}')

        os.chdir(base + root + f'EmE/EmE{p}_initS_1_linexam')
        modelEmE = MixtureOfExperts.models.load_model('EmE', f'EmE{p}_initS_1_linexam', root=os.getcwd() + '/')
        ec1, em1, es1 = errors(modelEmE, x_test[:, 0:p], burnin=1000, MC=np.min((5000, modelEmE._mcmcSteps)),
                               plot='save', thinning=10, returnE=True, loadDict=load)
        print(f'{ec1}, {em1}, {es1}')
base = "/l/gaddc1/Dropbox/"
root = "MixtureOfExperts/demos/santner/Isotropic/"

import sys, os
import numpy as np
import scipy.stats as stats
import MixtureOfExperts
from MixtureOfExperts.utils import simulate_data as sd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import pickle


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

def data_generating_function_density(xbins, ybins, mu1=2., mu2=6., tau1=1.2, tau2=1.2, coef1=0.1, coef2=-0.1,
                                     factor1=0.6, factor2=0.4):
    """
    Get the density (and plot) the true data generating function. Return density and
    """

    Xgrid, Ygrid = np.meshgrid(xbins, ybins)  # Grid for density (#y * #x)
    print(xbins.shape)
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

    #xbins = np.linspace(np.min(model.x), np.max(model.x), 200)
    ybins = np.linspace(np.min(model.y)*1.2, np.max(model.y)*1.2, 500)
    #Xgrid, Ygrid = np.meshgrid(xbins, ybins)

    # Make/get conditional predictions for expectation/density over (latent) response
    if loadDict is True:
        try:
            with open('density_dictionary.pkl', 'rb') as f:
                errorDict = pickle.load(f)
        except:
            raise RuntimeError('Unable to load marginal prediction dictionary')
    else:

        sampDict = plots_sample(model, Xtest, ybins, burnin=burnin, MC=MC, returnE=returnE, thinning=thinning)

        density_true = data_generating_function_density(xbins, ybins)
        density_samp_true = data_generating_function_density(sampDict['Xsamp'][:, 0], ybins)
        errorDict = {**sampDict, 'density_true':density_true, 'density_samp_true':density_samp_true}

        with open('density_dictionary.pkl', 'wb') as f:
            pickle.dump(errorDict, f, pickle.HIGHEST_PROTOCOL)

    # Unpack
    density_samp = errorDict['D_samp'][0]
    Xgrid_samp = errorDict['Xgrid_samp']
    Ygrid_samp = errorDict['Ygrid_samp']
    density_true = errorDict['density_true']
    density_samp_true = errorDict['density_samp_true']
    if {'E_cond', 'E_marg', 'E_samp'} <= errorDict.keys():
        expectation_samp = errorDict['E_samp'][0]
        Xsamp = errorDict['Xsamp'][:, 0]

    plt.figure()
    plt.imshow(density_samp)
    plt.figure()
    plt.imshow(density_samp_true)
    plt.show()

    # error_samp = np.linalg.norm(density_samp - density_samp_true) / np.linalg.norm(density_samp_true)
    if np.any(np.isnan(density_samp)):
        print("Warning: some NaN predictive densities. Setting density to 0")
        density_samp = np.nan_to_num(density_samp)
    error_samp = np.sum(np.abs(density_samp - density_samp_true)) / np.prod(np.shape(density_samp_true))

    if plot is not False:

        DensityYSlices = True
        Density = True
        Paper = True

        if DensityYSlices:
            plt.close('all')
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
            annotations = np.zeros((9, 10))

            for xi in range(10):
                area_samp = np.trapz(np.reshape(density_samp[:, 20 * xi], (1, -1)), dx=ybins[1] - ybins[0])
                area_true = np.trapz(np.reshape(density_true[:, 20 * xi], (1, -1)), dx=ybins[1] - ybins[0])

                ax1.plot(ybins, density_true[:, 20 * xi], label=area_true)
                ind = np.argmax(density_true[:, 20 * xi])
                annotations[0:3, xi] = [ybins[ind], density_true[ind, 20 * xi], area_true]

                ax2.plot(ybins, density_samp[:, 20 * xi], label=area_samp)
                ind = np.argmax(density_samp[:, 20 * xi])
                annotations[3:6, xi] = [ybins[ind], density_samp[ind, 20 * xi], area_samp]

            [ax1.annotate('{0:.1f}:{1:.2f}'.format(xbins[20 * xi], annotations[2, xi]), xy=annotations[0:2, xi], fontsize=7) for xi in range(10)]
            [ax2.annotate('{0:.1f}:{1:.2f}'.format(xbins[20 * xi], annotations[5, xi]), xy=annotations[3:5, xi], fontsize=8) for xi in range(10)]

            ax1.set_title('True')
            ax2.set_title('Sampled')

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
            plt.scatter(np.mean(model.x, axis=1), model.y, s=0.1)
            plt.title('Truth')

            plt.subplot(422)
            plt.pcolormesh(np.asarray(Xgrid_samp), Ygrid_samp, (density_samp_true), cmap='Reds')
            plt.colorbar()
            #plt.scatter(Xsamp[:, 0], np.zeros_like(Xsamp[:, 0]), s=0.1)
            plt.title('Truth - sampling')

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



    return error_samp


if __name__ == '__main__':
    """
    Generate the paper plots and metrics relevant to the predictive error
    """

    for p in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 125, 150, 200]:
        print(f'\nRunning error plots script for p={p}')

        # Pre-sample test points for fair comparison
        test_data = np.loadtxt(base + root + f"iso_santner{p}_mixture_test.csv", delimiter=", ")
        x_test = test_data[:, 0:-2]

        if p not in []:
            load = True
        else:
            load = False

        # Create coverage plots
        os.chdir(base + root + f'JmE/JmE{p}')
        modelJmE = MixtureOfExperts.models.load_model('JmE', f'JmE{p}', root=os.getcwd()+'/')
        js1 = errors(modelJmE, x_test[:, 0:p], burnin=100, MC=np.min((5000, modelJmE._mcmcSteps)),
                               plot=False, thinning=10, returnE=True, loadDict=load)
        print(f'{js1}')

        os.chdir(base + root + f'EmE/EmE{p}_initS_1_linexam')
        modelEmE = MixtureOfExperts.models.load_model('EmE', f'EmE{p}_initS_1_linexam', root=os.getcwd() + '/')
        es1 = errors(modelEmE, x_test[:, 0:p], burnin=1000, MC=np.min((5000, modelEmE._mcmcSteps)),
                               plot=True, thinning=10, returnE=True, loadDict=False)
        print(f'{es1}')
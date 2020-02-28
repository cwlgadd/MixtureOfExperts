import os
import pickle
import scipy.stats as stats
import MixtureOfExperts
from MixtureOfExperts.utils import simulate_data as sd
import matplotlib.pyplot as plt
import numpy as np
import logging

def true_density(xbins, ybins, mu1, mu2, tau1, tau2, coef1, coef2, factor1, factor2, noise):
    """
    Get the density (and plot) the true data generating function. Return density and
    """
    Xgrid, Ygrid = np.meshgrid(xbins, ybins)  # Grid for density (#y * #x)
    density_true = np.zeros_like(Xgrid)

    weights = sd.mixture(xbins, mu1=mu1, mu2=mu2, tau1=tau1, tau2=tau2)

    for xi in range(xbins.shape[0]):
        mean1 = np.exp(coef1 * xbins[xi]) * np.cos(factor1 * np.pi * xbins[xi])
        mean2 = np.exp(coef2 * xbins[xi]) * np.cos(factor2 * np.pi * xbins[xi])
        density_true[:, xi] += weights[xi] * stats.norm.pdf(ybins, loc=mean2, scale=noise[1])
        density_true[:, xi] += (1-weights[xi]) * stats.norm.pdf(ybins, loc=mean1, scale=noise[0])

    return density_true


def errors_test_samples(class_name, name, root, Xsamp, burnin=500, MC=None, thinning=1, returnE=False, load=False):

    if load is True:
        try:
            with open(root + 'predictive_test.pkl', 'rb') as f:
                dDict = pickle.load(f)
        except:
            raise RuntimeError('Unable to load dictionary for test sample predictive distribution ')
    else:
        # Load model
        model = MixtureOfExperts.models.load_model(class_name, name, root)
        print(model)
        if MC is None:
            MC = np.min((model._mcmcSteps, 5000))

        # The grid we want to calculate over
        #ybins = np.linspace(np.min(model.y) * 1.2, np.max(model.y) * 1.2, 1000)
        ybins = np.linspace(-6, 6, 1000)

        dDict = model._predict_density(Xsamp, y_test=ybins, burnin=burnin, MC=MC, thinning=thinning)
        dDict = {**dDict, 'Xsamp':Xsamp, 'ybins':ybins}
        if returnE is True:
            eDict = model._predict_expectation(Xsamp, burnin=burnin, MC=MC, thinning=thinning)
            #eDict['E_samp'] = eDict.pop('E')
            dDict = {**dDict, **eDict}
            plt.scatter(np.mean(Xsamp, axis=1), eDict["E"])
            plt.show()

        with open(root + 'predictive_test.pkl', 'wb') as f:
            pickle.dump(dDict, f, pickle.HIGHEST_PROTOCOL)

    density_samp = dDict['D'][0]
    ybins = dDict['ybins']

    density_samp_true = true_density(np.mean(Xsamp, axis=1), ybins, mu1=2., mu2=6., tau1=1.2, tau2=1.2,
                                     coef1=0.1, coef2=-0.1, factor1=0.6, factor2=0.4, noise=[0.15, 0.05])
    print(density_samp_true.shape)

    if np.any(np.isnan(density_samp)):
        print("Setting NaN/Inf density values to zero")
        density_samp = np.nan_to_num(density_samp)

    order = np.argsort(np.mean(Xsamp, axis=1))
    plt.figure()
    plt.imshow(density_samp[:, order])
    plt.colorbar()
    plt.figure()
    plt.imshow(density_samp_true[:, order])
    plt.colorbar()

    # for sample in range(Xsamp.shape[0]):
    #    print(dDict['D'][0].shape)
    #    plt.plot(ybins, dDict['D'][0][:, sample])
    #    plt.scatter(y_test[sample], 0)
    #    plt.show()

    error = np.sum(np.abs(density_samp - density_samp_true))
    l1_error_samp = (error * (ybins[2]-ybins[1])) / density_samp_true.shape[1]
    print(l1_error_samp)

    return l1_error_samp

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(module)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    date = 'Isotropic'
    name_jme = ''
    name_eme = '_initS_1_linexam'
    P = [5]  # [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 125, 150, 200]
    errorsJmE, errorsEmE = np.zeros(len(P)), np.zeros(len(P))
    # Plotting options
    plot = True#'save'                   # True to show, 'save' to save in model folder, or string to save in subfolder.
    movie = False                        # 'save' to save in model folder, or string to save in subfolder.

    for idx_p, p in enumerate(P):
        if p in []:
            load = False
        else:
            load = True
        # Load test data
        test_data = np.loadtxt(f"{date}/iso_santner{p}_mixture_test.csv", delimiter=", ")
        x_test = test_data[:, 0:-2]
        # y_test = test_data[:, -2]
        # label_test = test_data[:, -1]

        print("\n")
        errorsJmE[idx_p] = errors_test_samples('JmE', f'JmE{p}{name_jme}', f'{date}/JmE/JmE{p}{name_jme}/',
                                               x_test[:, 0:p], burnin=200, MC=None, thinning=5, returnE=False,
                                               load=load)
        errorsEmE[idx_p] = errors_test_samples('EmE', f'EmE{p}{name_eme}', f'{date}/EmE/EmE{p}{name_eme}/',
                                               x_test[:, 0:p], burnin=1000, MC=None, thinning=5, returnE=False,
                                                   load=load)

    plt.plot(P, errorsJmE)
    plt.show()

    print(errorsJmE)
    print(errorsEmE)
    # np.savetxt('L1errors.txt', np.vstack((errorsJmE, errorsEmE)), delimiter=', ')
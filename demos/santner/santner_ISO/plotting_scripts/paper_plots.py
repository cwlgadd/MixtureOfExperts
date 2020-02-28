import os
import pickle
import MixtureOfExperts
from MixtureOfExperts.utils import simulate_data as sd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import logging
from prettytable import PrettyTable


def data_generating_function_density(xbins, ybins, mu1=2., mu2=6., tau1=1.2, tau2=1.2, coef1=0.1, coef2=-0.1,
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


def data_generating_function_sample(x, mu1=3., mu2=5., tau1=0.8, tau2=0.8, coef1=0.1, coef2=-0.1, factor1=0.6,
                                    factor2=0.4):
    """

    """
    y = np.zeros_like(x)
    weights = sd.mixture(x, mu1=mu1, mu2=mu2, tau1=tau1, tau2=tau2)
    for xi in range(x.shape[0]):
        mean1 = np.exp(coef1 * x[xi]) * np.cos(factor1 * np.pi * x[xi])
        mean2 = np.exp(coef2 * x[xi]) * np.cos(factor2 * np.pi * x[xi])
        if weights[xi] > np.random.uniform(0, 1):
            y[xi] = stats.norm.rvs(loc=mean2, scale=0.05)
        else:
            y[xi] = stats.norm.rvs(loc=mean1, scale=0.15)

    #plt.scatter(x,y)
    #plt.show()

    return y



def plots_sample(model, Xsamp, ybins, burnin=500, MC=None, returnE=False, thinning=1):


    dDict = model._predict_density(Xsamp, y_test=ybins, burnin=burnin, MC=MC, thinning=thinning)
    dDict['D_samp'] = dDict.pop('D')
    dDict = {**dDict, 'Xsamp':Xsamp}
    if returnE is True:
        eDict = model._predict_expectation(Xsamp, burnin=burnin, MC=MC, thinning=thinning)
        eDict['E_samp'] = eDict.pop('E')
        dDict = {**dDict, **eDict}

    Xgrid_samp, Ygrid_samp = np.meshgrid(np.mean(Xsamp, axis=1), ybins)
    dDict = {**dDict, 'Xgrid_samp':Xgrid_samp, 'Ygrid_samp':Ygrid_samp}

    return dDict


def paper_plots(class_name, name, root, Xtest, burnin=500, MC=None, plot=False, thinning=1, returnE=False,
                        loadDict=False):
    """  """

    model = MixtureOfExperts.models.load_model(class_name, name, root)
    print(model)
    xbins = np.linspace(np.min(model.x), np.max(model.x), 200)
    ybins = np.linspace(np.min(model.y)*1.2, np.max(model.y)*1.2, 500)
    Xgrid, Ygrid = np.meshgrid(xbins, ybins)

    # Make/get conditional predictions for expectation/density over (latent) response
    if loadDict is True:
        try:
            with open(root + 'density_dictionary.pkl', 'rb') as f:
                errorDict = pickle.load(f)
        except:
            raise RuntimeError('Unable to load marginal prediction dictionary')
    else:

        sampDict = plots_sample(model, Xtest, ybins, burnin=burnin, MC=MC, returnE=returnE, thinning=thinning)

        # Get true densities
        density_true = data_generating_function_density(xbins, ybins)

        density_samp_true = data_generating_function_density(np.mean(sampDict['Xsamp'], axis=1), ybins)
        errorDict = {**sampDict, 'density_true':density_true, 'density_samp_true':density_samp_true}

        with open(root + 'density_dictionary.pkl', 'wb') as f:
            pickle.dump(errorDict, f, pickle.HIGHEST_PROTOCOL)

    for key in errorDict:
        print(key, 'corresponds to item with shape', np.shape(errorDict[key]))

    # Unpack
    # print(errorDict.keys())
    density_samp = errorDict['D_samp'][0]
    Xgrid_samp = errorDict['Xgrid_samp']
    Ygrid_samp = errorDict['Ygrid_samp']
    density_true = errorDict['density_true']
    density_samp_true = errorDict['density_samp_true']
    if {'E_samp'} <= errorDict.keys():
        expectation_samp = errorDict['E_samp'][0]
        Xsamp = errorDict['Xsamp']

    Density = True
    Paper = False

    if Density:
        plt.close('all')
        fontsize = 20
        s = 0.75
        c = 'k'
        plt.close('all')

        # Density sample truth
        fig = plt.figure() # plt.subplots(ncols=1) , (ax1)
        plt.pcolormesh(np.asarray(Xgrid_samp), Ygrid_samp, (density_samp_true), cmap='Reds')
        plt.colorbar()
        plt.scatter(np.mean(model.x, axis=1), model.y, s=s, c=c)
        plt.xlabel(r'$\bar{x}}$', fontsize=fontsize)
        plt.ylabel('$y$', fontsize=fontsize)
        plt.xticks([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlim((min(np.mean(Xtest, axis=1)), max(np.mean(Xtest, axis=1))))
        plt.ylim(min(ybins), max(ybins))
        plt.tight_layout()
        if plot is True:
            plt.show()
        elif plot is 'save':
            plt.savefig(root + 'density_sample_truth.eps', format='eps')
        elif isinstance(plot, str):
            plt.savefig(root + plot + 'density_sample_truth.eps', format='eps')
        else:
            raise ValueError
        plt.close('all')
        #plt.title('Truth - sampling')

        # Density sample model
        fig = plt.figure()  # plt.subplots(ncols=1) , (ax1)
        plt.pcolormesh(np.asarray(Xgrid_samp), Ygrid_samp, (density_samp), cmap='Reds')
        plt.colorbar()
        plt.scatter(np.mean(model.x, axis=1), model.y, s=s, c=c)
        if 'expectation_samp' in locals():
            plt.plot(np.mean(Xsamp, axis=1), expectation_samp, color=c)
        plt.xlabel(r'$\bar{x}}$', fontsize=fontsize)
        plt.ylabel('$y$', fontsize=fontsize)
        plt.xticks([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlim((min(np.mean(Xtest, axis=1)), max(np.mean(Xtest, axis=1))))
        plt.ylim(min(ybins), max(ybins))
        plt.tight_layout()
        if plot is True:
            plt.show()
        elif plot is 'save':
            plt.savefig(root + 'density_sample_model.eps', format='eps')
        elif isinstance(plot, str):
            plt.savefig(root + plot + 'density_sample_model.eps', format='eps')
        else:
            raise ValueError
        plt.close('all')
        #plt.title('Model sampling')

        error = np.sum(np.abs(density_samp - density_samp_true))
        l1_error_samp = (error * (ybins[2] - ybins[1])) / density_samp_true.shape[1]
        print(l1_error_samp)

        return l1_error_samp


def table_coverage(class_name, name, root, Xsamp, burnin=500, MC=None, plot=False, movie=False, thinning=1,
                   loadDict=False, CI='quant'):
    """

    :param class_name:
    :param name:
    :param root:
    :param burnin:
    :param MC:
    :param plot:
    :return:
    """
    plt.close('all')

    interval = 0.95
    model = MixtureOfExperts.models.load_model(class_name, name, root)
    #print(model)
    ybins = np.linspace(np.min(model.y) * 3, np.max(model.y) * 3, 500)

    # Make/get conditional predictions for expectation/density over (latent) response
    if loadDict is True:
        try:
            with open(root + 'coverage_dictionary.pkl', 'rb') as f:
                covDict = pickle.load(f)
        except:
            raise RuntimeError('Unable to load coverage dictionary')
    else:
        sampDict = plots_sample(model, Xsamp, ybins, burnin=burnin, MC=MC, thinning=thinning)
        y_samp = data_generating_function_sample(Xsamp[:, 0], mu1=3., mu2=5., tau1=0.8, tau2=0.8,
                                                 coef1=-0.2, coef2=-0.2, factor1=0.6, factor2=0.4)
        normaliser = np.sum(sampDict['D_samp'][0], axis=0)

        covDict = {**sampDict, 'y_samp': y_samp, 'normaliser': normaliser}

        with open(root + '/coverage_dictionary.pkl', 'wb') as f:
            pickle.dump(covDict, f, pickle.HIGHEST_PROTOCOL)

    #for key in covDict:
    #    print(key, 'corresponds to item with shape', np.shape(covDict[key]))

    # Unpack
    density_samp = covDict['D_samp'][0]
    y_samp = covDict['y_samp']
    normaliser = covDict['normaliser']
    Xsamp = covDict['Xsamp']

    #print(1/normaliser)
    #print(ybins[1]-ybins[0])
    #print(np.sum(density_samp[:, 0]/normaliser[0]))
    #print(np.cumsum(density_samp[:, 0]/normaliser[0]))

    bounds = [None] * Xtest.shape[0]
    if movie is not False:
        frame_counter = 0
    for xi in range(Xtest.shape[0]):
        norm_dens_xi = density_samp[:, xi]/normaliser[xi]
        cum_norm_dens_xi = np.cumsum(norm_dens_xi)
        if CI == 'quant':
            bounds[xi] = MixtureOfExperts.utils.quant(ybins, norm_dens_xi, interval=interval)
        elif CI == 'hpd':
            bounds[xi] = MixtureOfExperts.utils.hpd(ybins, norm_dens_xi, interval=interval)
        elif CI == 'hpd_union':
            bounds[xi] = MixtureOfExperts.utils.hpd_union(ybins, norm_dens_xi, interval=interval)
        else:
            raise ValueError('invalid interval')

        if movie is not False:
            if True: #not any(bounds[xi, 0] <= y_samp[xi] <= bounds[xi, 1]):            # add condition on frames shown
                scaled_norm_dens_xi = (norm_dens_xi / np.max(norm_dens_xi)) / 2
                plt.plot(ybins, scaled_norm_dens_xi)
                plt.plot(ybins, cum_norm_dens_xi)
                plt.scatter(y_samp[xi], 0)
                for bound in range(len(bounds[xi])):
                    plt.plot([bounds[xi][bound][0], bounds[xi][bound][0]], [0, 1], c='k')
                    plt.plot([bounds[xi][bound][1], bounds[xi][bound][1]], [0, 1], c='k')
                plt.title(Xsamp[xi, 0])
                if movie is not False:                                                  #TODO: fix conditioning of movie
                    fname = root + name + '/_tmp%06d.png' % frame_counter
                    frame_counter += 1
                    plt.savefig(fname)
                    plt.close()
                else:
                    plt.show()
                    plt.close()

    if movie is not False:
        if movie is not 'save':
            subfolder = movie
        else:
            subfolder = ''
        try:
            os.system("rm " + root + name + subfolder + "/coverage_movie_" + CI + ".mp4")
            print('Deleted old movie')
        except:
            pass
        os.system("ffmpeg -r " + str(5) + " -i " + root + name + "/_tmp%06d.png  " +
                  root + name + subfolder + "/coverage_movie_" + CI + ".mp4")
        os.system("rm " + root + name + "/_tmp*.png")
        print('Removed temporary frame files')

    count_in = 0
    widths = np.zeros((Xtest.shape[0]))
    for xi in range(Xtest.shape[0]):
        c = 'r'
        for bound in range(len(bounds[xi])):
            # coverage count
            if bounds[xi][bound][0] <= y_samp[xi] <= bounds[xi][bound][1]:
                c = 'b'
                count_in += 1
            # credible interval width
            widths[xi] += bounds[xi][bound][1] - bounds[xi][bound][0]

        if plot is not False:
            for bound in range(len(bounds[xi])):
                plt.plot(bounds[xi][bound] - y_samp[xi], [Xsamp[xi, 0], Xsamp[xi, 0]], c=c)

    if plot is not False:
        plt.xlim((-2,2))
    if plot is True:
        plt.show()
    elif plot is 'save':
        plt.savefig(root + "/coverage_" + CI + ".eps", format='eps')
    elif isinstance(plot, str):
        plt.savefig(root + plot + "/coverage_" + CI + ".eps", format='eps')
    else:
        raise ValueError

    coverage = count_in / Xtest.shape[0]

    plt.close('all')

    print('{0} out of {1}: {2}\%'.format(count_in, Xtest.shape[0], coverage))
    print('Average credible interval width {0}'.format(np.mean(widths)))

    return coverage, np.mean(widths)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(module)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    name_eme = '_initS_1_linexam'
    root_eme = '../Isotropic/EmE/EmE'

    name_jme = ''
    root_jme = '../Isotropic/JmE/JmE'

    # Which plots to demo
    coverage = False
    errors = True

    # Plotting options
    plot = 'save'                        # True to show, 'save' to save in model folder, or string to save in subfolder.
    movie = False                        # 'save' to save in model folder, or string to save in subfolder.

    # Create coverage plots
    if coverage:
        CI = ['hpd', 'hpd_union', 'quant']
        P = [1, 2, 5]
        t = PrettyTable(['Model', 'P', 'CI', 'CI95', 'mean CI95'], float_format='.1')
        results_jme = np.empty((len(CI)*len(P), 2))
        results_eme = np.empty((len(CI)*len(P), 2))
        counter_jme = 0
        counter_eme = 0
        for idc, ci in enumerate(CI):
            for idp, p in enumerate(P):
                results_jme[counter_jme, :] = \
                    table_coverage('JmE', 'JmE' + str(p) + name_jme, root_jme + str(p) + name_jme + '/', Xtest[:, 0:p],
                                   CI=ci, burnin=1000, MC=5000, plot=plot, movie=movie, thinning=50, loadDict=False)
                t.add_row(['JmE', p, CI[idc], '{0}'.format('%.4g' % results_jme[counter_jme, 0]),
                           '{0}'.format('%.4g' % results_jme[counter_jme, 1])])
                counter_jme += 1

                results_eme[counter_eme, :] = \
                    table_coverage('EmE', 'EmE' + str(p) + name_eme, root_eme + str(p) + name_eme + '/', Xtest[:, 0:p],
                                   CI=ci, burnin=1000, MC=5000, plot=plot, movie=movie, thinning=1, loadDict=True)
                t.add_row(['EmE', p, CI[idc], '{0}'.format('%.4g' % results_eme[counter_eme, 0]),
                           '{0}'.format('%.4g' % results_eme[counter_eme, 1])])
                counter_eme += 1
        print(t.get_string(title="Santner coverage"))

    # Create predictive density plots
    if errors:
        P = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 125, 150, 200]
        joint_errors = np.zeros(len(P))
        enriched_errors = np.zeros(len(P))
        for idx_p, p in enumerate(P):
            test_data = np.loadtxt(f"../Isotropic/iso_santner{p}_mixture_test.csv", delimiter=", ")
            Xtest = test_data[:, 0:-2]
            Xtest = Xtest[np.argsort(np.mean(Xtest, axis=1)), :]

            js1 = paper_plots('JmE', 'JmE' + str(p) + name_jme, root_jme + str(p) + name_jme + '/', Xtest[:, 0:p],
                              burnin=100, MC=None, plot=plot, thinning=100, returnE=False, loadDict=False)

            es1 = paper_plots('EmE', 'EmE' + str(p) + name_eme, root_eme + str(p) + name_eme + '/', Xtest[:, 0:p],
                              burnin=1000, MC=5000, plot=plot, thinning=100, returnE=False, loadDict=False)
            joint_errors[idx_p] = js1
            enriched_errors[idx_p] = es1

        print(joint_errors)
        print(enriched_errors)
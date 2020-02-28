base = "/l/gaddc1/Dropbox/"
root = "MixtureOfExperts/demos/santner/santner_ARD/ARD/Apr15_fromTriton/"
CI = ['hpd', 'hpd_union', 'quant']
interval = 0.95

import sys, os
sys.path.insert(0, base)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import MixtureOfExperts
from MixtureOfExperts.utils import simulate_data as sd
import pickle
from scipy import stats


def sample_true(x, mu1=3., mu2=5., tau1=0.8, tau2=0.8, coef1=0.1, coef2=-0.1, factor1=0.6, factor2=0.4, samples=30):
    y = np.zeros((x.shape[0], samples))
    weights = sd.mixture(x, mu1=mu1, mu2=mu2, tau1=tau1, tau2=tau2)
    for xi in range(x.shape[0]):
        mean1 = np.exp(coef1 * x[xi]) * np.cos(factor1 * np.pi * x[xi])
        mean2 = np.exp(coef2 * x[xi]) * np.cos(factor2 * np.pi * x[xi])
        for s in range(samples):
            if weights[xi] > np.random.uniform(0, 1):
                y[xi, s] = stats.norm.rvs(loc=mean2, scale=0.05)
            else:
                y[xi, s] = stats.norm.rvs(loc=mean1, scale=0.15)
    #for i in range(samples):
    #    plt.scatter(x, y[:, i])
    #plt.show()
    return y

def density_model(model, x_test, ybins, burnin=500, MC=None, thinning=1):

    density_dict = model._predict_density(x_test, y_test=ybins, covariate=0, burnin=burnin, MC=MC, thinning=thinning)
    density_dict['D_samp'] = density_dict.pop('D')
    density_dict = {**density_dict, 'Xsamp':x_test}
    Xgrid_samp, Ygrid_samp = np.meshgrid(x_test[:, 0], ybins)
    dDict = {**density_dict, 'Xgrid_samp':Xgrid_samp, 'Ygrid_samp':Ygrid_samp}

    return dDict

def coverage(model, x_test, burnin=500, MC=None, plot=False, movie=False, thinning=1, loadDict=False, CI='quant', s=10):

    ybins = np.linspace(np.min(model.y) * 3, np.max(model.y) * 3, 500)
    frame_counter = 0

    # Make/get conditional predictions for expectation/density over (latent) response
    if loadDict is True:
        try:
            with open('coverage_dictionary.pkl', 'rb') as f:
                coverage_dict = pickle.load(f)
        except:
            raise RuntimeError('Unable to load coverage dictionary')
    else:
        y_samp = sample_true(x_test[:, 0], samples=s)
        density_dict = density_model(model, x_test, ybins, burnin=burnin, MC=MC, thinning=thinning)
        coverage_dict = {**density_dict,
                         'y_samp': y_samp,
                         'normaliser': np.sum(density_dict['D_samp'][0], axis=0)}

        with open('coverage_dictionary.pkl', 'wb') as f:
            pickle.dump(coverage_dict, f, pickle.HIGHEST_PROTOCOL)

    # Unpack
    density_samp = coverage_dict['D_samp'][0]
    y_samp = coverage_dict['y_samp']
    normaliser = coverage_dict['normaliser']
    Xsamp = coverage_dict['Xsamp']

    bounds = [None] * x_test.shape[0]

    for xi in range(x_test.shape[0]):
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

        # Plot frames of the movie and save to file.
        # ... These show the pred density, the credible interval sets, and the true samples for each test point
        if movie is not False:
            #if not any(bounds[xi, 0] <= y_samp[xi] <= bounds[xi, 1]):            # add condition on frames shown
            scaled_norm_dens_xi = (norm_dens_xi / np.max(norm_dens_xi)) / 2
            plt.plot(ybins, scaled_norm_dens_xi)
            plt.plot(ybins, cum_norm_dens_xi)
            plt.scatter(y_samp[xi, :], np.zeros(y_samp.shape[1]), s=2)
            for bound in range(len(bounds[xi])):
                plt.plot([bounds[xi][bound][0], bounds[xi][bound][0]], [0, 1], c='k')
                plt.plot([bounds[xi][bound][1], bounds[xi][bound][1]], [0, 1], c='k')
            plt.title(f'x_1 = {Xsamp[xi, 0]}')
            plt.xlim((-3, 3))
            fname = '_tmp%06d.png' % frame_counter
            frame_counter += 1
            plt.savefig(fname)
            plt.close()

    # Create the movie
    if movie is not False:
        # If a previous movie exists for this type of credible interval, then delete
        try:
            os.system("rm coverage_movie_" + CI + ".mp4")
            print('Deleted old movie')
        except:
            pass
        # Create movie then delete the temporary files
        os.system("ffmpeg -r " + str(2) + " -i _tmp%06d.png coverage_movie_" + CI + ".mp4")
        os.system("rm _tmp*.png")
        print('Removed temporary frame files')

    # Get the coverage and credible interval sizes
    count_in = 0
    widths = np.zeros((x_test.shape[0]))
    for xi in range(x_test.shape[0]):
        for bound in range(len(bounds[xi])):
            # credible interval width
            widths[xi] += bounds[xi][bound][1] - bounds[xi][bound][0]

        # coverage count
        for i in range(y_samp.shape[1]):
            c = 'r'
            for bound in range(len(bounds[xi])):
                if bounds[xi][bound][0] <= y_samp[xi, i] <= bounds[xi][bound][1]:
                    c = 'b'
                    count_in += 1

            if plot is not False:
                for bound in range(len(bounds[xi])):
                    plt.plot(bounds[xi][bound] - y_samp[xi, i], [Xsamp[xi, 0], Xsamp[xi, 0]], c=c)

    coverage = count_in / (x_test.shape[0] * y_samp.shape[1])

    if plot is not False:
        plt.xlim((-3.5, 3.5))
        plt.ylabel('$x_1$')
        plt.xlabel('credible interval (centered)')
        plt.tight_layout()

    if plot is True:
        plt.show()
    elif plot is 'save':
        plt.savefig(f"{interval}coverage{coverage:.2f}_avgwidth{np.mean(widths):.2f}_" + CI + ".eps", format='eps')
    else:
        raise ValueError
    plt.close('all')

    print('{0} out of {1}: {2}\%'.format(count_in, x_test.shape[0]*y_samp.shape[1], coverage))
    print('Average credible interval width {0}'.format(np.mean(widths)))

    return coverage, np.mean(widths)


if __name__ == '__main__':
    """
    Generate the paper plots and metrics relevant to the coverage
    """

    ci = sys.argv[1]
    assert ci in CI

    test_data = np.loadtxt(base + root + f"santner_mixture_test_new.csv", delimiter=", ")
    x_test = test_data[:, 0:-2]

    for p in [1, 2, 5, 10, 15, 20]:
        print(f'\n Running coverage plots script for p={p} and credible interval \"{ci}\"')

        # Create coverage plots
        os.chdir(base + root + f'JmE/JmE{p}_initS_linexam')
        modelJmE = MixtureOfExperts.models.load_model('JmE', f'JmE{p}_initS_linexam', root=os.getcwd()+'/')
        coverage(modelJmE, x_test[:, 0:p], CI=ci, burnin=1000, MC=5000, plot='save', movie=False, thinning=10,
                 loadDict=True, s=5)

        os.chdir(base + root + f'EmE/EmE{p}_initS_1_linexam')
        modelEmE = MixtureOfExperts.models.load_model('EmE', f'EmE{p}_initS_1_linexam', root=os.getcwd()+'/')
        coverage(modelEmE, x_test[:, 0:p], CI=ci, burnin=1000, MC=5000, plot='save', movie=False, thinning=10,
                 loadDict=True, s=5)
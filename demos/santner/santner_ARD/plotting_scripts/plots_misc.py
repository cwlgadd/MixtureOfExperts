import MixtureOfExperts
import numpy as np
import logging
import sys


def model_xlikelihood(class_name, name, root):

    model = MixtureOfExperts.models.load_model(class_name, name, root)
    model._xlikelihood.plot_xlikelihood(model.x)


def model_priors(class_name, name, root):

    model = MixtureOfExperts.models.load_model(class_name, name, root)
    model._expert.plot_hyperpriors(separate=False)


def model_posterior(class_name, name, root, burnin=500, MC=None):

    model = MixtureOfExperts.models.load_model(class_name, name, root)
    MixtureOfExperts.plotting.plot_posteriors(model, start=burnin, stop=MC, save=root + name + '_')


def model_movie(class_name, name, root, burnin=500, MC=None, auto=False, path=''):

    model = MixtureOfExperts.models.load_model(class_name, name, root)
    MixtureOfExperts.plotting.make_movie(model, start=burnin, stop=MC, auto=auto, path=path)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(module)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    date = 'Aug16'
    name_jme = '_initS_linexam'
    name_eme = '_initS_1_linexam'
    p = 1#sys.argv[1]

    # Plotting options
    plot = True#'save'                   # True to show, 'save' to save in model folder, or string to save in subfolder.
    movie = False                        # 'save' to save in model folder, or string to save in subfolder.

    # Load test data
    test_data = np.loadtxt(f"../{date}/santner_mixture_test.csv", delimiter=", ")
    x_test = test_data[:, 0:-2]
    y_test = test_data[:, -2]
    label_test = test_data[:, -1]

    # Create model plots

    # x-likelihood
    model_xlikelihood('JmE', f'JmE{p}{name_jme}', f'../{date}/JmE/JmE{p}{name_jme}/')
    model_xlikelihood('EmE', f'EmE{p}{name_eme}', f'../{date}/EmE/EmE{p}{name_eme}/')

    # Hyper-priors
    model_priors('JmE', f'JmE{p}{name_jme}', f'../{date}/JmE/JmE{p}{name_jme}/')
    model_priors('EmE', f'EmE{p}{name_eme}', f'../{date}/EmE/EmE{p}{name_eme}/')

    # Hyper-posteriors
    model_posterior('JmE', f'JmE{p}{name_jme}', f'../{date}/JmE/JmE{p}{name_jme}/', burnin=0, MC=None)
    model_posterior('EmE', f'EmE{p}{name_eme}', f'../{date}/EmE/EmE{p}{name_eme}/', burnin=0, MC=None)

    # Gibbs movie
    #model_movie('JmE', f'JmE{p}{name_jme}', f'{date}/JmE/JmE{p}{name_jme}/', burnin=0, MC=None, auto=True,
    #            path = f'../{date}/JmE/JmE{p}{name_jme}/')
    #model_movie('EmE', f'EmE{p}{name_eme}', f'{date}/EmE/EmE{p}{name_eme}/', burnin=0, MC=None, auto=True,
    #             path = f'../{date}/EmE/EmE{p}{name_eme}/')

base = "/l/gaddc1/Dropbox/"

import numpy as np
import GPy
import logging, os, sys
sys.path.insert(0, base + 'MixtureOfExperts/')
import MixtureOfExperts

def demo_EmE(X, Y, DPiters=1000):
    """
    Explore the posterior of our model.
    """
    print(X.shape)
    """ Create expert shell """
    expert = MixtureOfExperts.experts.IndpendentRBF(input_dim=X.shape[1], output_dim=Y.shape[1],
                                                    process_mean='Constant', ard=False)
    expert._prior_mean = GPy.priors.Gaussian(0, 0.5)
    expert.lengthscale = GPy.priors.Gamma.from_EV(1.25 * np.sqrt(p), 1.5 * np.sqrt(X.shape[1]))
    expert.kernel_variance = GPy.priors.Gamma.from_EV(3 / (np.log(p) + 1), 0.5)
    expert.likelihood_variance = GPy.priors.LogGaussian(np.log(0.01), 1)

    """ Create xlikelihood """
    xlikelihood = MixtureOfExperts.input_models.NormalInverseGamma(np.mean(X), 0.25, 2, 1)

    """ Create model shell - or loads if pickled file exists """
    model = MixtureOfExperts.models.EnrichedMixtureOfExperts(X, Y, expert, xlikelihood=xlikelihood,
                                                             name='EmE' + str(np.shape(X)[1]) + '_initS_1_linexam',
                                                             mneal8=3, init_f='singleton', init_c=1, SM=True) #, 2
    print(model)

    """ Run chain """
    model(samples=DPiters)

    """ Save progress """
    model.save()
    logging.info(model)


    return model


def demo_pull_information(model, root):
    """
    Create files for allocation matrices to calculate posterior similarity matrix
    Create some plots
    """

    indexes_f, indexes_c = model.indexes
    with open(root + 'EmE/' + model.__name__ + "/feature.csv", "wb") as f:
        np.savetxt(f, indexes_f.astype(int), fmt="%i", delimiter=", ")
    with open(root + 'EmE/' + model.__name__ + "/covariate.csv", "wb") as f:
        np.savetxt(f, indexes_c.astype(int), fmt="%i", delimiter=", ")

if __name__ == "__main__":
    """
    Demo for the Enriched Mixture of Experts model.
    """
    #assert sys.version_info >= (3, 6)
    p = int(sys.argv[1])
    print(f'Running Santner demo for the Enriched Mixture of Experts model')

    # Set root
    root = "MixtureOfExperts/demos/santner/santner_ISO/Isotropic/"

    # Start/re-start logging
    logpath = base + root + f'EmE/EmE{p}_initS_1_linexam/console.log'
    print(f'logging to {logpath}')
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(module)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG, filename=logpath)    #

    # Load data
    Data = np.loadtxt(base+root+f"iso_santner{p}_mixture_train.csv", delimiter=", ")
    print(Data.shape)
    T = np.reshape(Data[:, -1], (-1, 1))
    Y = np.reshape(Data[:, -2], (-1, 1))
    X = Data[:, 0:-2]
    assert p == X.shape[1]

    # Run Joint Mixture of Experts model in blocks
    for _ in range(200):
        Xp = X[:, 0:p]

        if p == 1:
            Xp = np.reshape(Xp, (-1, 1))

        model = demo_EmE(Xp, Y, DPiters=25)
        demo_pull_information(model, base+root)



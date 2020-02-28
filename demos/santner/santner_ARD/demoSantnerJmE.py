base = "/l/gaddc1/Dropbox/"

import numpy as np
import GPy
import logging, os, sys
sys.path.insert(0, base + 'MixtureOfExperts/')
import MixtureOfExperts


def demo_JmE(X, Y, DPiters=1000):
    """
    Explore the posterior of our model.
    """
    #assert np.shape(X)[1] <= 8
    print(os.getcwd())

    """ Create expert shell """
    expert = MixtureOfExperts.experts.IndpendentRBF(input_dim=X.shape[1], output_dim=Y.shape[1],
                                                    process_mean='Constant', ard=True)
    expert._prior_mean = GPy.priors.Gaussian(0, 0.5)
    expert.lengthscale = [GPy.priors.Gamma(10, 1/2)] * X.shape[1]       # 15, 1/3
    expert.lengthscale[0] = GPy.priors.Gamma(3, 1)          
    print(expert.lengthscale[1].summary())
    expert.kernel_variance = GPy.priors.Gamma(2, 1.5)
    expert.likelihood_variance = GPy.priors.LogGaussian(np.log(0.01), 0.5)

    """ Create xlikelihood """
    xlikelihood = MixtureOfExperts.input_models.NormalInverseGamma(np.mean(X), 0.25, 2, 1)

    """ Create model shell - or loads if pickled file exists """
    model = MixtureOfExperts.models.JointMixtureOfExperts(X, Y, expert, xlikelihood=xlikelihood,
                                                          name='JmE' + str(np.shape(X)[1]) + '_initS_linexam',
                                                          mneal8=3, initialisation='singleton')
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

    indexes_f = model.indexes
    with open(root + 'JmE/' + model.__name__ + "/feature.csv", "wb") as f:
        np.savetxt(f, indexes_f.astype(int), fmt="%i", delimiter=", ")


if __name__ == "__main__":
    """
    Demo for the Joint Mixture of Experts model.
    """
    #assert sys.version_info >= (3, 6)
    p = int(sys.argv[1])
    print(f'Running Santner demo for the Joint Mixture of Experts model')

    # Set root
    root = "MixtureOfExperts/demos/santner/santner_ARD/ARD/"

    # Start/re-start logging
    logpath = base + root + f'JmE/JmE{p}_initS_linexam/console.log'
    print(f'logging to {logpath}')
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(module)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG, filename=logpath)    #

    # Load data
    Data = np.loadtxt(base + root + "santner_mixture_new.csv", delimiter=", ")
    T = np.reshape(Data[:, -1], (-1, 1))
    Y = np.reshape(Data[:, -2], (-1, 1))
    X = Data[:, 0:-2]
    assert p <= X.shape[1]

    # Run Joint Mixture of Experts model in blocks
    for _ in range(200):
        Xp = X[:, 0:p]

        if p == 1:
            Xp = np.reshape(Xp, (-1, 1))

        model = demo_JmE(Xp, Y, DPiters=25)
        demo_pull_information(model, base + root)


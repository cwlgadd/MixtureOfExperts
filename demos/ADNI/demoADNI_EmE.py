import numpy as np
import pickle
import MixtureOfExperts
import GPy
import csv
import logging
import os

def demo_EmE(X, Y, DPiters=1000):
    """
    Explore the posterior of our model.
    """

    """ Create expert shell """
    expert = MixtureOfExperts.experts.IndpendentRBF(input_dim=X.shape[1], output_dim=Y.shape[1],
                                                    process_mean='Constant', ard=True)
    expert._prior_mean = GPy.priors.Gaussian(20, 7.5)
    expert.lengthscale = [GPy.priors.Gamma(3, 1),            # MMSE
                          GPy.priors.Gamma(3, 0.15),         # Age
                          GPy.priors.Gamma(2, 5),            # Gender
                          GPy.priors.Gamma(5, 1),            # Education
                          GPy.priors.Gamma(3, 5),            # APOE
                          GPy.priors.Gamma(2, 4)]            # Diagnosis
    expert.kernel_variance = GPy.priors.Gamma(2, 1)
    expert.likelihood_variance = GPy.priors.Gamma(1.5, 0.5)

    """ Create mixed likelihood """
    BBlikelihood_MMSE = MixtureOfExperts.input_models.BetaBinomial(Gp=30, gammap=[5, 1])
    BBlikelihood_EDU = MixtureOfExperts.input_models.BetaBinomial(Gp=20, gammap=[3, 2])
    BBlikelihood_APOE = MixtureOfExperts.input_models.BetaBinomial(Gp=2, gammap=[1, 3])
    BBlikelihood_DIAG = MixtureOfExperts.input_models.BetaBinomial(Gp=3, gammap=[1, 1])
    DMlikelihood = MixtureOfExperts.input_models.DirichletMultinomial(gammap=np.ones((2, 1)))
    Clikelihood = MixtureOfExperts.input_models.NormalInverseGamma(72, 2, 2, 10)
    mixedLikelihood = MixtureOfExperts.input_models.MixedInput(likelihoods=[BBlikelihood_MMSE, Clikelihood,
                                                                                 DMlikelihood, BBlikelihood_EDU,
                                                                                 BBlikelihood_APOE, BBlikelihood_DIAG],
                                                                    indexes=[[0], [1], [2], [3], [4], [5]])

    """ Create probit model """
    probit = MixtureOfExperts.link_functions.orderedProbit(expert, L=30, minmax=[0, 29])
    print(probit)
    print(probit.categories)

    """ Create model shell - or loads if pickled file exists """
    model = MixtureOfExperts.models.EnrichedMixtureOfExperts(X, Y, expert, mixedLikelihood, probit=probit,
                                                             name='EmE_adni_initbaseline_1', mneal8=3,
                                                             init_f=X[:, -1], init_c=1)
    print(model)

    """ Run chain """
    model(samples=DPiters)

    """ Save progress """
    model.save()
    print(model)

    return model


def demo_pull_information(model, root, burnin=150):
    """
    Create files for allocation matrices to calculate posterior similarity matrix
    Create some plots
    """

    indexes_f, indexes_c = model.indexes
    with open(root + 'EmE/' + model.__name__ + "_feature.csv", "wb") as f:
        np.savetxt(f, indexes_f.astype(int), fmt="%i", delimiter=", ")
    with open(root + 'EmE/' + model.__name__ + "_covariate.csv", "wb") as f:
        np.savetxt(f, indexes_c.astype(int), fmt="%i", delimiter=", ")

if __name__ == "__main__":
    """
    Demo for the Enriched Mixture of Experts model.
    """
    #assert sys.version_info >= (3, 6)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(module)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)  # filename='adniE.log',

    X = np.zeros((767, 6))
    Y = np.zeros((767, 1))

    """ Load data """
    base = "/l/gaddc1/Dropbox (Aalto)/"
    root = "MixtureOfExperts/demos/ADNI/ADNI/"
    with open(base + root + "ADNI_Training_Q1_APOE_July22.2014.csv", 'rU') as f:
        reader = csv.reader(f)
        counter = 0
        counter_in = 0
        counter_out = 0
        for row in reader:
            if counter == 0:
                pass
            else:
                if counter % 2 != 0:  # if counter is odd, baseline
                    # MMSE - ordinal
                    X[counter_in, 0] = row[9]
                    # age - continuous
                    X[counter_in, 1] = row[5]
                    # gender - categorical discrete
                    if row[6] == 'Male':
                        X[counter_in, 2] = 1
                    # education = ordinal
                    X[counter_in, 3] = row[7]
                    # genotype = ordinal
                    X[counter_in, 4] = row[8]
                    # baseline diagnosis
                    if row[4] == 'CN':
                        X[counter_in, 5] = 0
                    elif row[4] == 'EMCI':
                        X[counter_in, 5] = 1
                    elif row[4] == 'LMCI':
                        X[counter_in, 5] = 2
                    elif row[4] == 'AD':
                        X[counter_in, 5] = 3
                    else:
                        raise ValueError

                    counter_in += 1
                else:
                    Y[counter_out] = row[9]
                    counter_out += 1
            counter += 1

    # subsample before preprocessing if we want to do that
    Xtrain = X[0::2, :]
    Ytrain = Y[0::2, :]
    Xtest = X[1::2, :]
    Ytest = Y[1::2, :]
    print(np.unique(Xtrain[:, -1]))

    data = [Xtrain, Ytrain, Xtest, Ytest]

    with open(base + root + 'data.pkl', 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

    """ Run Joint Mixture of Experts model for different choices of #covariate inputs """
    for _ in range(94):
        model = demo_EmE(Xtrain, Ytrain, DPiters=50)
        demo_pull_information(model, root=base+root, burnin=0)

import GPy
import numpy as np
import GPy.inference.mcmc.hmc as hmca

def metropoliswrapper(model, Ntotal=10000, Nburn=1000, Nthin=10, tune=True, tune_throughout=False, tune_interval=10):
    assert Ntotal > Nburn, 'Need more samples than burn-in'

    mhmc = GPy.inference.mcmc.Metropolis_Hastings(model)
    mhmc.sample(Ntotal=Ntotal, Nburn=Nburn, Nthin=Nthin, tune=tune, tune_throughout=tune_throughout,
                tune_interval=tune_interval)
    chains = mhmc.chains

    smhmc = np.zeros((len(chains[0]), len(chains[0][0])))
    for sample in range(len(chains[0])):
        smhmc[sample, :] = chains[0][sample]

    return smhmc


def hmcshortcutwrapper(model, hmcsamples, M=None, stepsize_range=[1e-6, 1e-1], groupsize=5, Hstd_th=[1e-5, 3.]): #[1e-6, 1e-1]

    hmc = hmca.HMC_shortcut(model, M=M, stepsize_range=stepsize_range, groupsize=groupsize, Hstd_th=Hstd_th)
    shmc = hmc.sample(m_iters=hmcsamples)

    return shmc


def hmcwrapper(model, hmcsamples, M=None, stepsize=2e-2):

    hmc = GPy.inference.mcmc.HMC(model, M=M, stepsize=stepsize)
    shmc = hmc.sample(num_samples=hmcsamples)

    return shmc



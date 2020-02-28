# MixtureOfExperts
Code for enriched mixtures of experts in Python. Stable release.

<!--- * [homepage](https://github.com/cwlgadd/MixtureOfExperts/)
-->

## Citation

    @Misc{GaddMoE2018,
      author =   {{Charles Gadd}},
      title =    {{MixtureOfExperts}: A package for enriched infinite mixtures of Gaussian process experts in python},
      howpublished = {\url{http://gitlab.com/charles1992/MixtureOfExperts}},
      year = {since 2018}
    }

## Installing 

If you'd like to install from source, or want to contribute to the code (i.e. by sending pull requests via github)
then:

    $ git clone https://gitlab.com/charles1992/MixtureOfExperts.git 
    $ cd MixtureOfExperts
    $ git checkout devel
    $ python setup.py build_ext --inplace

## Demos / Reproducing results

Scripts to reproduce the results in the related paper can be found in the demos folder.
* Set working directory to the model folder, e.g. EmE5_initS_1_linexam.
* Demos include the .csv files holding the cluster allocation samples from posterior sampling. 
* However, pre-trained classes are not shared due to space restrictions. These can be made available on request for the santner example.

Santner (mixtures of dampened cosines):
* Two data generating models considered here. The first demonstrates the improved flexibility of the enriched prior, the second demonstrates the improved scaling with input dimension D. 
* The santner scripts require you to specify the number of covariates as a script argument. 
* Data for these examples can be found in the subfolder. 

ADNI (Alzheimer's disease Neuroimaging initiative):
* This example demonstrates how to use a mixture of different input types and an ordinal output.
* Data for this example cannot be made public and must be requested from the initiative.

Notebooks:
* Coming soon...

## Notes on dependencies

* The current GP expert class uses GPy. Different experts can be added fairly trivially (for example if you want to add sparsity or use a different Bayesian model).
* The probit models require either emcee or rpy2 to sample from the truncated multivariate distributions.
    * rpy2 is quicker (Gibbs sampling), but not trivial to install on some systems.
    * rpy2 is used by default, and emcee (rejection sampling) can be used if not available. However, rejection sampling may lead to poor samples if the sampling space is high dimensional. It is strongly recommended to use rpy2!


<!---
 * Experts
    * experts/: A script demonstrating the functionality of the experts class.
    * (TODO) likelihoods/: A script to demonstrate the functionality of the input models.  

-->

<!---
## TODO:

* Docker container
* Replace relative paths
* Fix input models other than Gaussian broken from re-naming methods
* Add base classes
* Potentially change module hierarchy.
    * MCMC moves classes to be split, moved to models/inference/split_merge_[model_name].py
    * link_functions, moved to experts/link_functions
    * plotting, moved to utils/plotting
-->


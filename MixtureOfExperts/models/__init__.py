from .joint_mixture_of_experts import JointMixtureOfExperts
from .enriched_mixture_of_experts import EnrichedMixtureOfExperts
import logging
logger = logging.getLogger(__name__)
import os


def load_model(class_name, name, root=os.getcwd()+'/'):
    import numpy, logging, sys, MixtureOfExperts
    empty_data = numpy.zeros((10, 1))
    empty_expert = MixtureOfExperts.experts.IndpendentRBF(1,1)

    if (class_name == 'JointMixtureOfExperts') or (class_name == 'JmE'):
        empty_model = JointMixtureOfExperts(empty_data, empty_data, empty_expert, None, name=name)
        empty_model.load(root=root)
        logger.info('Loaded Joint Mixture of Experts model from root directory {0}, continuing'.format(root))
    elif (class_name == 'EnrichedMixtureOfExperts') or (class_name == 'EmE'):
        empty_model = EnrichedMixtureOfExperts(empty_data, empty_data, empty_expert, None, name=name)
        empty_model.load(root=root)
        logger.info('Loaded Enriched Mixture of Experts model from root directory {0}, continuing'.format(root))
    else:
        logging.critical('Unable to load model, exiting.')
        sys.exit()

    return empty_model

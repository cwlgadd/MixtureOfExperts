import MixtureOfExperts
import numpy as np
import matplotlib.pyplot as plt
import os, sys


def load_enriched(name):

    model = MixtureOfExperts.models.EnrichedMixtureOfExperts(np.ones((1, 1)), np.ones((1, 1)), None, None, name=name)
    # shouldn't reach this as the way we load will raise an exception if it can't load. TODO: add better loading
    assert model._mcmcSteps > 0, 'Was unable to load, ensure you are in the same directory.'

    return model


def load_joint(name):

    model = MixtureOfExperts.models.JointMixtureOfExperts(np.ones((1, 1)), np.ones((1, 1)), None, None, name=name)
    # shouldn't reach this as the way we load will raise an exception if it can't load. TODO: add better loading
    assert model._mcmcSteps > 0, 'Was unable to load, ensure you are in the same directory.'

    return model


def make_movie(model, start=0, stop=None, auto=False, path=''):
    """ Make movie out of a Markov Chain
        This function creates and deletes _many_ temporary files. """

    print('Making movie for ' + path + model.__name__)
    if auto is False:
        while True:
            try:
                # Note: Python 2.x users should use raw_input, the equivalent of 3.x's input
                response = input("This function generates (then deletes) _many_ temporary files in directory " +
                                 os.getcwd() + ". \n Type `yes` to continue:")
                assert response == 'yes'
            except:
                pass
            else:
                break

    if stop == None:
        stop = model._mcmcSteps - 1

    counter = 0
    for state in np.linspace(start, stop, stop-start+1):
        print('Creating frame {0} of {1}'.format(int(state+1), int(stop+1)))
        MixtureOfExperts.plotting.prediction.state_summary(model, state=int(state), GPs=True, mean=False,
                                                           density=False, covariate=0, feature=0)
        fname = '_tmp%06d.png' % counter
        counter += 1

        plt.savefig(fname)
        plt.close()

    try:
        os.system("rm " + path + model.__name__ + ".mp4")
        print('Deleted old movie')
    except:
        pass

    os.system("ffmpeg -r " + str(5) + " -i _tmp%06d.png  " + path + model.__name__ + ".mp4")

    os.system("rm _tmp*.png")
    print('Removing temporary frame files')

    return 1


if __name__ == '__main__':
    # must demo from directory model is saved in.


    name = 'EmE5_init6_6'
    model = load_joint(name)
    make_movie(model)


from __future__ import division
import numpy as np
from sklearn import preprocessing as skpp


__all__ = ['pre', 'post', '_remove_constant', '_add_constant']

def pre(matrix):
    """
    Take the training data and put everything needed to undo this operation later into a dictionary.

    :param      matrixTrain:
    :return:    matrixTrainPost:
                preprocDict:
    """
    preprocDict = {}

    # constants
    matrix, index, constant = _remove_constant(matrix)
    preprocDict.update({'index': index, 'constant': constant})

    # scale
    scaler = skpp.StandardScaler().fit(matrix)
    matrix = scaler.transform(matrix)
    preprocDict.update({'scaler': scaler})

    return matrix, preprocDict


def post(matrix, preprocDict):
    """
    Given a reduced matrix, find the full and unnormalised matrix.

    :param      matrixPost:
                preprocDict:
    :return:
    """

    # scale
    scaler = preprocDict['scaler']
    matrix = scaler.inverse_transform(matrix)

    # constants
    matrix = _add_constant(matrix, preprocDict['index'], preprocDict['constant'])

    return matrix


def _remove_constant(matrix):
    """
    Remove constant features.

    :param      matrix:         matrix with constant features
    :return:    matrixR:        matrix with constant features removed
                index:          vector of indexes where True == keep, False == constant and removed.
                constants:      matrix of constants removed for adding back later
    """

    index = (np.var(matrix, 0) != 0)

    matrix_r = np.zeros([np.shape(matrix)[0], sum(index)])
    i = 0
    for n in range(len(index)):
        if index[n]:
            matrix_r[:, i] = matrix[:, n]
            i += 1

    constants = matrix[0, np.invert(index)]

    return matrix_r, index, constants


def _add_constant(matrix, index, constants):
    """
    Add constant features back in.

    :param      matrix:          matrix with constant features removed
                index:           vector of indexes where True == kept, False == constant and removed.
                constants:       vector of constants previously removed
    :return:    matrixF          matrix with constant features added back in.
    """

    # tile the constants for each data point
    constants = np.matlib.repmat(constants, np.shape(matrix)[0], 1)

    # add constants back in to a full matrix
    matrixF = np.zeros((np.shape(matrix)[0], np.shape(index)[0]))
    matrixF[:, index] = matrix
    matrixF[:, np.invert(index)] = constants

    return matrixF



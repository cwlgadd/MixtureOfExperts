from __future__ import division
import numpy as np
import copy

__all__ = ['quant', 'hpd', 'hpd_union']

def quant(bins, bins_density, interval=0.95):
    """
    Get the indexes defining range of equal tail density (for one interval) given binned density values.

    :param bin_density:
    :return: indexLower, indexUpper:         the indexes corresponding to the edge of the interval
        :type       list of one tuple
    """
    assert bins.shape == bins_density.shape
    assert np.isclose(np.sum(bins_density), 1)

    cum_dens = np.cumsum(bins_density)
    lower = np.sum(cum_dens < (1 - interval) / 2)
    upper = np.sum(cum_dens < (1 + interval) / 2)

    # print(cum_dens[np.sum(cum_dens < (1 - interval) / 2) - 1])
    # print(cum_dens[np.sum(cum_norm_dens_xi < (1 - interval) / 2)])
    # print(cum_dens[np.sum(cum_dens < (1 - interval) / 2) + 1])
    # print(cum_dens[np.sum(cum_dens < (1 + interval) / 2) - 1])
    # print(cum_dens[np.sum(cum_dens < (1 + interval) / 2)])
    # print(cum_dens[np.sum(cum_dens < (1 + interval) / 2) + 1])

    return [(bins[lower], bins[upper])]


def hpd(bins, bins_density, interval=0.95):
    """
    Get the indexes defining range of highest posterior density (for one interval) given binned density values.

    :param bin_density:
    :return: indexLower, indexUpper:         the indexes corresponding to the edge of the interval
        :type       list of one tuple
    """
    assert bins.shape == bins_density.shape
    assert np.isclose(np.sum(bins_density), 1)

    left_dens_index = np.argmax(bins_density)                           # the initial left range index
    right_dens_index = left_dens_index                                  # the initial right range index
    CIprob = bins_density[left_dens_index]                              # the initial CI area
    bounds = [bins[left_dens_index], bins[right_dens_index]]            # the initial CI interval

    error_counter = 0
    while CIprob < interval:
        if error_counter > len(bins):
            raise ValueError('We should not need more iterations than bins')

        # Propose expanding to either side if valid index
        prop_left_dens = 0
        prop_right_dens = 0
        if left_dens_index - 1 >= 0:
            prop_left_dens = bins_density[left_dens_index - 1]
        if right_dens_index + 1 <= len(bins) - 1:
            prop_right_dens = bins_density[right_dens_index + 1]

        # Choose a proposal based on which has highest posterior density
        if prop_right_dens > prop_left_dens:
            CIprob += prop_right_dens
            bounds[1] = bins[right_dens_index + 1]
            right_dens_index += 1
        else:
            CIprob += prop_left_dens
            bounds[0] = bins[left_dens_index - 1]
            left_dens_index -= 1

    return [(bounds[0], bounds[1])]


def hpd_union(bins, bins_density, interval=0.95):
    """
    Get the indexes defining range of the union of highest posterior density intervals given binned density values.

    :param bin_density:
    :return:
        :type       list of tuples
    """
    assert bins.shape == bins_density.shape
    assert np.isclose(np.sum(bins_density), 1), bins_density

    bins_density = copy.deepcopy(bins_density)                           # we don't want to alter in code called from

    next_highest_ind = np.argmax(bins_density)
    indexes = [next_highest_ind]
    CIprob = bins_density[next_highest_ind]                              # the initial CI area
    bins_density[next_highest_ind] = 0

    error_counter = 0
    while CIprob < interval:
        if error_counter > len(bins):
            raise ValueError('We should not need more interations than bins')

        next_highest_ind = np.argmax(bins_density)
        indexes.append(next_highest_ind)
        CIprob += bins_density[next_highest_ind]  # the initial CI area
        bins_density[next_highest_ind] = 0
    indexes.sort()

    pairs = list()
    for i in range(len(indexes)-1):
        if indexes[i+1] - indexes[i] == 1:
            pairs.append((indexes[i], indexes[i+1]))

    indices = []
    for begin, end in sorted(pairs):
        if indices and indices[-1][1] >= begin - 1:
            indices[-1] = (indices[-1][0], end)
        else:
            indices.append((begin, end))

    intervals = list()
    for i in range(len(indices)):
        intervals.append((bins[indices[i][0]], bins[indices[i][1]]))

    return intervals

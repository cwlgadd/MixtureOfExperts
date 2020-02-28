from __future__ import division
from MixtureOfExperts.utils.misc import exp_normalize as exp_norm
from MixtureOfExperts.utils.misc import sum_lnspace as sum_ln
import scipy.stats
import math
import numpy as np
import random
import copy
import logging
logger = logging.getLogger(__name__)

__all__ = ['SplitMerge']

class SplitMerge(object):
    """
    A class for the split-merge (tailored version of dumb+smart/smart+dumb moves, see Wang and Russell (2015))
    """

    # public (accessible through @property below)
    _model = None                                       # model we we are performing the moves on.

    # private (used internally)
    _model_type = None                                  # Type of model we we are performing the moves on.

    # Enriched model specific parameters.               TODO:  Split joint and enriched up with inheritance.
    _kx1plus = None                                     # The number of x clusters with more than one data point
                                                        #  (the number that can be split)
    _kx1plus_indexes = None
    _kx2plus = None                                     # The number of x-clusters within each y-clusters containing 
                                                        #   more than one x-cluster (number that can be merged)
    _kx2plus_indexes = None

    # Joint model specific parameters

    @property
    def model(self):
        """
        Get the model
        """
        return self._model

    @property
    def sIndex(self):
        return self._model._sIndex

    @sIndex.setter
    def sIndex(self, indices):
        self._model._sIndex = indices

    @property
    def xlik(self):
        return self.model._xlikelihood

    def __init__(self, model):
        """
        Initialize the split/merge class.

        :param x:               Observed features
            :type               np.ndarray (#samples * #features)
        :param y:               Observed covariates
            :type               np.ndarray (#samples * #covariates)
        """
        self._model = model
        self._model_type = model.__class__.__name__
        #print(self._model_type)

        self._kx1plus, self._kx1plus_indexes = model.kx1plus
        self._kx2plus, self._kx2plus_indexes = model.kx2plus
        self.__name__ = f'SplitMerge{model.__name__}'

    def __call__(self, *args, **kwargs):
        """
        Perform the dumb+smart or smart+dumb split merge move.
        """

        if self._model_type == 'EnrichedMixtureOfExperts':
            # Enriched mixture model
            self._move_enriched()
        # elif self._model_type == 'JointMixtureOfExperts':
        #    # Joint mixture model
        #    self._move_joint()
        else:
            raise NotImplementedError('Split-merge is not implemented for that model type.')

        # Check the model has not been broken from this move (all parameters are consistent with each other)
        self.model.check_consistent(last_update='after split/merge move')

        return self._model

    def _move_joint(self):
        """
        Perform dumb+smart or smart+dumb split merge move for the joint mixture model
        :return:
        """
        assert self._model_type == 'JointMixtureOfExperts',\
            f'Cannot perform enriched model\'s split-merge on a model of class {self.model.__class__}'

        # smart split / dumb merge move
        if np.random.uniform(0, 1, 1) > 0.5:
            self._joint_smartsplit()
        else:
            self._joint_dumbsplit()

        return

    def _joint_smartsplit(self):
        raise NotImplementedError('Smart split is not implemented for the joint mixture model')

    def _joint_dumbsplit(self):
        raise NotImplementedError('Dumb split is not implemented for the joint mixture model')

    def _move_enriched(self):
        """
        Perform dumb+smart or smart+dumb split merge move for the enriched mixture model
        :return:
        """
        assert self._model_type == 'EnrichedMixtureOfExperts', \
            f'Cannot perform enriched model\'s split-merge on a model of class {self.model.__class__}'

        # smart split / dumb merge move
        if np.random.uniform(0, 1, 1) > 0.5:
            if self._kx1plus > 1:
                self._enriched_smartsplit()
            else:
                logging.info(f'Smart-split impossible, all x-clusters are singletons. kx1+={self._kx1plus}')
        else:
            if self._kx2plus > 1:
                self._enriched_dumbmerge()
            else:
                logging.info(f'Dumb-merge impossible, all y-clusters contain only one x-cluster. kx2+={self._kx2plus}')

        # dumb split / smart merge move
        if np.random.uniform(0, 1, 1) > 0.5:
            if self._kx1plus > 1:
                self._enriched_dumbsplit()
            else:
                logging.info(f'Dumb-split impossible, all x-clusters are singletons. kx1+ {self._kx1plus}')
        else:
            if self._kx2plus > 1:
                self._enriched_smartmerge()
            else:
                logging.info(f'Smart-merge impossible, all y-clusters contain only one x-cluster. kx2+ {self._kx2plus}')

        return

    def _enriched_smartsplit(self):
        """
        Perform the smart split move for the enriched model.
        """

        # Select an x-cluster to split from the kx1+ x-clusters containing more than one data point.
        log_probability = np.zeros((len(self._kx1plus_indexes),1))
        for count, [i, j] in enumerate(self._kx1plus_indexes, 0):                          # Loop over kx1+ clusters
            xlj = self.model.x[(self.sIndex[:, 0] == i) & (self.sIndex[:, 1] == j)]        # Get the inputs from it
            log_probability[count] = -self.xlik.log_marginal(xlj)                          # Calculate the likelihood
            assert np.isinf(log_probability[count]) == False, f'{log_probability[count]}, {self.xlik.log_marginal(xlj)}'
            assert xlj.shape[0] > 1                                                        # Check contains more than 1
        cumsum = np.cumsum(exp_norm(log_probability))                                      # Norm-exp and cumsum
        apbool_select = scipy.stats.uniform(0, 1).rvs(1) > cumsum                          # Choose x-cluster to split
        idx_split = self._kx1plus_indexes[np.min(np.where(apbool_select == 0))]            # Get cluster index to split
        idx_new = [idx_split[0], np.max(self.sIndex[self.sIndex[:, 0] == idx_split[0], 1]) + 1] # index of proposed
        logger.debug(f'Smart-split. Considering splitting {idx_split} to new {idx_new}')

        # Record original cluster
        xlj_orig, idx_xlj_orig = self.model.subcluster(idx_split[0], idx_split[1])         # samples in cluster to split

        # propose split, i.e. new x-cluster allocations
        sIndex_prop = copy.copy(self.sIndex)
        probability_seqprob = np.nan * np.ones((len(idx_xlj_orig), 2))      # Memory-alloc for move probabilities
        for count, idx_n in enumerate(idx_xlj_orig):                        # For each sample in cluster to be split
            sIndex_n = sIndex_prop[:idx_n, :]                               # The indices up to (and excluding) idx_n
            x_n = self.model.x[:idx_n, :]                                   # ... and corresponding samples

            # Points (so far) proposed to stay (excluding points loop hasn't reached)
            x_stay = x_n[(sIndex_n[:, 0] == idx_split[0]) & (sIndex_n[:, 1] == idx_split[1]), :]

            # Points (so far) proposed to move (excluding points loop hasn't reached)
            x_moved = x_n[(sIndex_n[:, 0] == idx_new[0]) & (sIndex_n[:, 1] == idx_new[1]), :]

            # Probability sample n in sub-cluster stays in the same sub-cluster
            if x_stay.shape[0] > 0:
                # If we have previously proposed samples to stay then evaluate the predictive
                log_prob_stay = self.xlik.log_predictive(self.model.x[[idx_n], :], x_stay)
            else:
                # Else currently no samples previously proposed to stay, get probability using marginal
                log_prob_stay = self.xlik.log_marginal(self.model.x[[idx_n], :])

            # Probability sample n in sub-cluster moves to new cluster
            if x_moved.shape[0] > 0:
                # If we have previously proposed samples to move then evaluate the predictive
                log_prob_move = self.xlik.log_predictive(self.model.x[[idx_n], :], x_moved)
            else:
                # Else currently no samples previously proposed to move, so evaluate marginal
                log_prob_move = self.xlik.log_marginal(self.model.x[[idx_n], :])

            # choose proposal based on probabilities
            probability_seqprob[[count], :] = exp_norm(np.asarray([log_prob_stay, log_prob_move])).T     # Normalise and exp
            apbool = scipy.stats.uniform(0, 1).rvs(1) > np.cumsum(probability_seqprob[count, :])    # n_i's proposal
            if np.min(np.where(apbool == 0)) == 0:
                # Stay. No need to update proposal indices
                logger.debug(f"Smart-split sample {idx_n}, stay proposed with p = {probability_seqprob[count, 0]} ")
            else:
                # Move, update proposal indices
                sIndex_prop[idx_n, :] = idx_new
                logger.debug(f"Smart-split sample {idx_n}, move proposed with p = {probability_seqprob[count, 1]}")

        # Metropolis accept/reject proposal step
        num_stay = np.count_nonzero((sIndex_prop[:, 0] == idx_split[0]) & (sIndex_prop[:, 1] == idx_split[1]))
        if num_stay == idx_xlj_orig.shape[0]:
            # If entire sequence proposed to stay then no move is performed (equivalent to accepting with prob 1)
            logger.info(f"Smart-split. All samples proposed to stay, doing nothing")
        elif num_stay == 0:
            # If entire sequence proposed to move then no move is performed (equivalent to accepting with prob 1)
            logger.info(f"Smart-split. All samples proposed to move, doing nothing")
        else:
            x_stay = self.model.x[(sIndex_prop[:, 0] == idx_split[0]) & (sIndex_prop[:, 1] == idx_split[1]), :]
            x_move = self.model.x[(sIndex_prop[:, 0] == idx_new[0]) & (sIndex_prop[:, 1] == idx_new[1]), :]

            # Calculate first term in p
            log_term1 = np.log(self.model._alpha_psi[int(idx_split[0])]) + \
                        math.lgamma(int(x_stay.shape[0])) + \
                        math.lgamma(int(x_move.shape[0])) + \
                        self.xlik.log_marginal(x_move) + \
                        self.xlik.log_marginal(x_stay) - \
                        math.lgamma(int(xlj_orig.shape[0])) - \
                        self.xlik.log_marginal(xlj_orig)

            # Calculate second term in p
            #kj = len(np.unique(self.sIndex[(self.sIndex[:, [0]] == idx_split[0])[:, 0], 1])) # num x clusters in split y
            kj = len(np.unique(self.sIndex[self.sIndex[:, 0] == idx_split[0], 1]))
            if kj == 1:                                                                  # If one unique, then kx2+
                kx2plus_prop = self._kx2plus + 2                                         # ... will increase by 2
            else:                                                                        # otherwise
                kx2plus_prop = self._kx2plus + 1                                         # ... will increases by 1
            log_term2 = -np.log(kx2plus_prop) - np.log(kj)                               # Second fraction of p

            # Calculate third term in p
            log_term3 = np.log(np.sum(np.exp(log_probability))) + self.xlik.log_marginal(xlj_orig)
            # log_term3 = -np.log(exp_norm(log_probability)[np.min(np.where(apbool_select == 0)), 0])      # equivalent

            # Calculate fourth term in p. Each factor of the product is the inverse normalised sequential probability
            log_term4 = 0
            for count, idx_n in enumerate(idx_xlj_orig):
                # Product over samples xlj_orig
                if np.all(sIndex_prop[idx_n, :] == idx_new):
                    # If sample corresponding to this factor was proposed to move then...
                    log_factor = -np.log(probability_seqprob[count, 1])
                    #print(f'\t probs {probability_seqprob[count, :]} to new with log factor {log_factor}')
                else:
                    # Else sample corresponding to this factor was proposed to stay then...
                    log_factor = -np.log(probability_seqprob[count, 0])
                    #print(f'\t probs {probability_seqprob[count, :]} to same with log factor {log_factor}')

                log_term4 += log_factor #np.log(term4_numerator) - term4_log_denominator
                #print(np.log(term4_numerator) - term4_log_denominator)

            p = np.exp(log_term1 + log_term2 + log_term3 + log_term4)
            ap = np.min((1, p))

            logger.info(f'Smart-split: p={p}, log terms ({log_term1} {log_term2} {log_term3} {log_term4})')
            apbool = (scipy.stats.uniform(0, 1).rvs(1) < ap )
            if apbool:                                                                      # accept, update self.model
                logger.info(f'Smart-split move accepted with probability {ap}, p={p}')
                logger.info(f'\t ... split {idx_split}, creating {idx_new}')
                # Update model
                self.sIndex = sIndex_prop
                self.model._kl_counter[int(idx_split[0])] += 1#\
                                #len(np.unique(self.sIndex[self.sIndex[:,0]==idx_split[0], 1]))
                # Update self
                self._kx1plus, self._kx1plus_indexes = self.model.kx1plus
                self._kx2plus, self._kx2plus_indexes = self.model.kx2plus
            else:
                logger.info(f'Smart-split move rejected with probability {ap}, p={p}')

    def _enriched_dumbmerge(self):
        """
        Perform the dumb merge move for the enriched model.
        """

        # select an x-cluster to merge uniformly from the kx2+ x-clusters in a y-cluster with more than one x-cluster.
        idx_lj = random.choice(self._kx2plus_indexes)                               # Sub-cluster to split from kx2+

        # Choose second cluster to merge from uniformly from the remaining x-clusters
        ltildes = (np.unique(self.sIndex[self.sIndex[:, 0] == idx_lj[0], 1]))       # x-clusters in samples y-cluster
        ltildes = list(filter((idx_lj[1]).__ne__, ltildes))                         # filter already chosen x-cluster
        idx_ljtilde = [idx_lj[0], random.choice(ltildes)]                           # make choice uniformly
        xlj, idx_xlj = self.model.subcluster(idx_lj[0], idx_lj[1])                  # Get samples and their indices
        xljtilde, idx_xljtilde = self.model.subcluster(idx_ljtilde[0], idx_ljtilde[1])  # ... for each selection

        # Proposal
        idx_xmerge = np.sort(np.hstack((idx_xlj, idx_xljtilde)))                    # Indicies in the proposed cluster
        x_merged = self.model.x[idx_xmerge, :]                                      # ... and their samples
        sIndex_prop = copy.copy(self.sIndex)                                        # Proposed indices
        sIndex_prop[(self.sIndex[:, 0] == idx_lj[0]) & (self.sIndex[:, 1] == idx_lj[1]), 1] = idx_ljtilde[1]
        #print(f'proposing merge of {idx_lj} and {idx_ljtilde}')

        # Metropolis accept/reject proposal step
        #   ... first term
        log_term1 = math.lgamma(int(x_merged.shape[0]))+ \
                    self.xlik.log_marginal(x_merged) - \
                    np.log(self.model._alpha_psi[int(idx_lj[0])]) - \
                    math.lgamma(int(xlj.shape[0])) - \
                    math.lgamma(int(xljtilde.shape[0])) - \
                    self.xlik.log_marginal(xlj) - \
                    self.xlik.log_marginal(xljtilde)

        #  ... second term
        kj = len(np.unique(self.sIndex[(self.sIndex[:, [0]] == idx_lj[0])[:, 0], 1])) # num of x-clusters in y-cluster j
        log_term2 = np.log(self._kx2plus) +  np.log(kj - 1)

        #  ... third term.
        log_term3_numerator = -self.xlik.log_marginal(x_merged)
        log_term3_denominator = -np.inf                                      # Start sum at 0 (in log space)
        H = np.unique(sIndex_prop, axis=0)                                   # All non-empty sub-clusters in proposal
        for h in H:                                                          #  ... loop over them
            xltjt = self.model.x[(sIndex_prop[:, 0] == h[0]) & (sIndex_prop[:, 1] == h[1]), :]      # covariates in h
            log_term3_denominator = sum_ln(log_term3_denominator, -self.xlik.log_marginal(xltjt))
            #print(f'index {h} updated log denom to {log_term3_denominator} with contribution of {-self.xlik.log_marginal(xltjt)}')
        log_term3 = log_term3_numerator - log_term3_denominator
        assert np.isfinite(log_term3)
        #print(f'log term 3: {log_term3_numerator} - {log_term3_denominator}')

        #  ... fourth term.
        log_term4 = 0
        for count, idx_n in enumerate(idx_xmerge):
            # For each sample in proposed merged cluster

            idx_xmerge_n = idx_xmerge[:count]              # Indicies of samples (before current idx_n) in merged
            x_merge_n = self.model.x[idx_xmerge_n, :]      # ... and corresponding samples in merged
            sIndex_merge_n = self.sIndex[idx_xmerge_n, :]  # ... and their original allocations, with samples below
            x_lj_n = x_merge_n[(sIndex_merge_n[:, 0] == idx_lj[0]) & (sIndex_merge_n[:, 1] == idx_lj[1]), :]
            x_ltj_n = x_merge_n[(sIndex_merge_n[:, 0] == idx_ljtilde[0]) & (sIndex_merge_n[:, 1] == idx_ljtilde[1]), :]

            # The 1st term in the denominator (predictive marginal for idx_n given those that were originally in l|j)
            if len(x_lj_n) == 0:
                log_stay = self.xlik.log_marginal(self.model.x[[idx_n]])
            else:
                log_stay = self.xlik.log_predictive(self.model.x[[idx_n]], x_lj_n)

            # The 2nd term in the denominator (predictive marginal for idx_n given those that were originally in l~|j)
            if len(x_ltj_n) == 0:
                log_move = self.xlik.log_marginal(self.model.x[[idx_n]])
            else:
                log_move = self.xlik.log_predictive(self.model.x[[idx_n]], x_ltj_n)

            # The numerator
            if sIndex_prop[idx_n, 1] == idx_lj[1]:                       # if nth sample (idx_n) proposed to l|j
                log_factor = log_stay - sum_ln(log_stay, log_move)
            elif sIndex_prop[idx_n, 1] == idx_ljtilde[1]:                # if nth sample (idx_n) proposed to l~|j
                log_factor = log_move - sum_ln(log_stay, log_move)
            else:
                raise RuntimeError('Should not be reachable')

            # Update product with new factor based on n
            log_term4 += log_factor

        p = np.exp(log_term1 + log_term2 + log_term3 + log_term4)
        ap = np.min((1, p))

        logger.info(f'Dumb-merge: p={p}, log terms ({log_term1} {log_term2} {log_term3} {log_term4})')

        apbool = (scipy.stats.uniform(0, 1).rvs(1) < ap)
        if apbool:                                                                  # accept, update
            logger.info(f'Dumb-merge move accepted with probability {ap}, p={p}')
            logger.info(f'\t ... merging {idx_ljtilde} into {idx_lj}')

            # Update model
            self.sIndex[idx_xljtilde, 1] = idx_lj[1]
            #  ... no new clusters created so we don't need to update cluster counters.

            # Update self
            self._kx1plus, self._kx1plus_indexes = self.model.kx1plus
            self._kx2plus, self._kx2plus_indexes = self.model.kx2plus
        else:
            logger.info(f'Dumb-merge move rejected with probability {ap}, p={p}')

        #raise NotImplementedError('Dumb merge is not implemented for the joint mixture model')

    def _enriched_dumbsplit(self):
        """
        Perform the dumb split move for the enriched model.
        """

        # Select an x-cluster to split uniformly from the kx1+ x-clusters containing more than one sample
        idx_split = random.choice(self._kx1plus_indexes)                                        # Sub-cluster to split
        xlj_orig, idx_xlj_orig = self.model.subcluster(idx_split[0], idx_split[1])              # ... and the samples

        # Cluster to split to
        idx_new = [idx_split[0], np.max(self.sIndex[self.sIndex[:, 0] == idx_split[0], 1]) + 1] # Proposed x-cluster
        logger.debug(f'Dumb-split. Considering splitting {idx_split} to new {idx_new}')

        # Proposal
        split_choice = np.random.choice(a=[False, True], size=(len(idx_xlj_orig),), p=[0.5, 1-0.5])
        sIndex_prop = copy.copy(self.sIndex)
        sIndex_prop[idx_xlj_orig[split_choice], 1] = idx_new[1]

        # Metropolis accept/reject proposal step
        if np.sum(split_choice) == len(split_choice):
            logger.info(f"Dumb-split. All {len(split_choice)} samples in sub-cluster proposed to move, doing nothing")
        elif np.sum(split_choice) == 0:
            logger.info(f"Dumb-split. All {len(split_choice)} samples in sub-cluster proposed to stay, doing nothing")
        else:
            # ... samples of new clusters under proposal
            x_stay = self.model.x[(sIndex_prop[:, 0] == idx_split[0]) & (sIndex_prop[:, 1] == idx_split[1]), :]
            x_move = self.model.x[(sIndex_prop[:, 0] == idx_new[0]) & (sIndex_prop[:, 1] == idx_new[1]), :]

            # Calculate first term in p
            log_term1 = np.log(self.model._alpha_psi[int(idx_split[0])]) + \
                        math.lgamma(int(x_stay.shape[0])) + \
                        math.lgamma(int(x_move.shape[0])) + \
                        self.xlik.log_marginal(x_stay) + \
                        self.xlik.log_marginal(x_move) - \
                        math.lgamma(int(xlj_orig.shape[0])) - \
                        self.xlik.log_marginal(xlj_orig)

            # Calculate second term in p
            log_term2 = np.log(self._kx1plus) + (xlj_orig.shape[0] - 1)*np.log(2)

            # Calculate third term in p
            #kj = len(np.unique(self.sIndex[(self.sIndex[:, [0]] == idx_split[0])[:, 0], 1]))# num x clusters in split y
            kj = len(np.unique(self.sIndex[self.sIndex[:, 0] == idx_split[0], 1]))
            if kj == 1:                                                                      # If only one unique, then
                kx2plus_prop = self._kx2plus + 2                                             # ... will increase by 2
            else:                                                                            # otherwise
                kx2plus_prop = self._kx2plus + 1                                             # ... it increases by 1
            log_term3 = -np.log(kx2plus_prop) - np.log(kj)

            # Calculate fourth term in p.
            log_numerator =  self.xlik.log_marginal(xlj_orig)                                # shared numerator of terms
            # ... part 1 denominator
            term4_part1_denominator = 0                                                      # Memory-allocation
            H = np.unique(sIndex_prop[(sIndex_prop[:, 0] == idx_split[0]) & (sIndex_prop[:, 1] != idx_split[1]), :],
                          axis=0)
            for h in H:
                x_h = self.model.x[(sIndex_prop[:, 0] == h[0]) & (sIndex_prop[:, 1] == h[1]), :]
                term4_part1_denominator += np.exp(self.xlik.log_marginal(np.vstack((x_stay, x_h))))
            log_term4_part1 = log_numerator - np.log(term4_part1_denominator)
            # ... part 2 denominator
            term4_part2_denominator = 0
            H = np.unique(sIndex_prop[(sIndex_prop[:, 0] == idx_new[0]) & (sIndex_prop[:, 1] != idx_new[1]), :],
                          axis=0)
            for h in H:
                x_h = self.model.x[(sIndex_prop[:, 0] == h[0]) & (sIndex_prop[:, 1] == h[1]), :]
                term4_part2_denominator += np.exp(self.xlik.log_marginal(np.vstack((x_move, x_h))))
            log_term4_part2 = log_numerator - np.log(term4_part2_denominator)
            # ... combine
            log_term4 = np.log( np.exp(log_term4_part1) + np.exp(log_term4_part2) )

            p = np.exp(log_term1 + log_term2 + log_term3 + log_term4)
            ap = np.min((1, p))
            logger.info(f'Dumb-split: p={p}, log terms ({log_term1} {log_term2} {log_term3} {log_term4})')

            apbool = (scipy.stats.uniform(0, 1).rvs(1) < ap)
            if apbool:  # accept, update self.model
                logger.info(f'Dumb-split move accepted with probability {ap}, p={p}')
                logger.info(f'\t ... split {idx_split}, creating {idx_new}')
                # Update model
                self.sIndex = sIndex_prop
                self.model._kl_counter[int(idx_split[0])] += 1
                # Update self
                self._kx1plus, self._kx1plus_indexes = self.model.kx1plus
                self._kx2plus, self._kx2plus_indexes = self.model.kx2plus
            else:
                logger.info(f'Dumb-split move rejected with probability {ap}, p={p}')

            #raise NotImplementedError('Dumb-split is not implemented for the enriched mixture model')

    def _enriched_smartmerge(self):
        """
        Perform the smart merge move for the enriched model.
        """

        # select an x-cluster to merge uniformly from the kx2+ x-clusters in a y-cluster with more than one x-cluster.
        idx_cluster1 = random.choice(self._kx2plus_indexes)                       # Sub-cluster to split from kx2+
        x1, idx_x1 = self.model.subcluster(idx_cluster1[0], idx_cluster1[1])      # ... and samples + indices

        # Choose second cluster to merge from the remaining x-clusters with probability prop to marginal of merged
        H = np.unique(self.sIndex[(self.sIndex[:, 0] == idx_cluster1[0]) &
                                  (self.sIndex[:, 1] != idx_cluster1[1]), :], axis=0)
        log_probability = np.zeros((H.shape[0],))                                        # Memory allocation
        for count, h in enumerate(H):
            xh = self.model.x[(self.sIndex[:, 0] == h[0]) &
                              (self.sIndex[:, 1] == h[1]), :]
            log_probability[count] = self.xlik.log_marginal(np.vstack((x1, xh)))         # Marginal for combined cluster
        probability_select = exp_norm(log_probability)                                   # Normalise and exp
        apbool_select = scipy.stats.uniform(0, 1).rvs(1) > np.cumsum(probability_select) # Sample index of cluster to...
        idx_cluster2 = H[np.min(np.where(apbool_select == 0)), :]                        # ... move to
        x2, idx_x2 = self.model.subcluster(idx_cluster2[0], idx_cluster2[1])             # x in l'|j

        # Proposal
        idx_x12 = np.sort(np.hstack((idx_x1, idx_x2)))                                 # Indices in the proposed cluster
        x12 = self.model.x[idx_x12, :]                                                 # ... and their samples
        sIndex_prop = copy.copy(self.sIndex)                                           # Proposed indices, move l to l'
        sIndex_prop[(self.sIndex[:, 0] == idx_cluster1[0]) &                           # Re-allocate under proposal
                    (self.sIndex[:, 1] == idx_cluster1[1]), 1] = idx_cluster2[1]       # ... putting 1 into 2

        # Metropolis accept/reject proposal step
        #   ... first term
        log_term1 = math.lgamma(int(x12.shape[0])) + \
                    self.xlik.log_marginal(x12) - \
                    np.log(self.model._alpha_psi[int(idx_cluster1[0])]) - \
                    math.lgamma(int(x1.shape[0])) - \
                    math.lgamma(int(x2.shape[0])) - \
                    self.xlik.log_marginal(x1) - \
                    self.xlik.log_marginal(x2)
        #  ... second term
        if (x1.shape[0] == 1) and (x2.shape[0] == 1):                                   # If two singletons merged
            proposed_kx1plus = self._kx1plus + 1                                        # ... will increase by 1
        elif (x1.shape[0] == 1) or (x2.shape[0] == 1):                                  # If one merged is singleton
            proposed_kx1plus = self._kx1plus                                            # ... no change
        else:                                                                           # If neither are singletons
            proposed_kx1plus = self._kx1plus - 1                                        # ... will decrease by 1
        log_term2 = -np.log(proposed_kx1plus)
        #  ... third term
        log_term3 = -(x12.shape[0] -1)*np.log(2)
        #  ... fourth term
        log_term4 = np.log(self._kx2plus)
        #  ... fifth term
        log_numerator = self.xlik.log_marginal(x12)                                     # shared numerators
        # ... ... part 1 denominator
        H = np.unique(self.sIndex[(self.sIndex[:, 0] == idx_cluster1[0]) &              # In the y-cluster
                                  (self.sIndex[:, 1] != idx_cluster1[1]), :], axis=0)   # ... but not in x-cluster 1
        log_term5_part1_denominator = -np.inf                                           # Initialise sum
        for h in H:
            x_h, idx_h1 = self.model.subcluster(h[0], h[1])
            log_term5_part1_denominator = sum_ln(log_term5_part1_denominator,
                                                 self.xlik.log_marginal(np.vstack((x1, x_h))))
        log_term5_part1 = log_numerator - log_term5_part1_denominator
        # ... ... part 2 denominator
        H = np.unique(self.sIndex[(self.sIndex[:, 0] == idx_cluster2[0]) &              # In the y-cluster
                                  (self.sIndex[:, 1] != idx_cluster2[1]), :], axis=0)   # ... but not in x-cluster 2
        log_term5_part2_denominator = -np.inf                                           # Initialise sum
        for h in H:
            x_h, idx_h1 = self.model.subcluster(h[0], h[1])
            log_term5_part2_denominator = sum_ln(log_term5_part2_denominator,
                                                 self.xlik.log_marginal(np.vstack((x2, x_h))))
        log_term5_part2 = log_numerator - log_term5_part2_denominator
        # ... ... combine part 1 and part 2
        log_term5 = -sum_ln(log_term5_part1, log_term5_part2)

        # Acceptance probability
        p = np.exp(log_term1 + log_term2 + log_term3 + log_term4 + log_term5)
        ap = np.min((1, p))
        logger.info(f'Smart-merge: p={p}, log terms ({log_term1} {log_term2} {log_term3} {log_term4} {log_term5})')
        # ... and accept/reject
        apbool = (scipy.stats.uniform(0, 1).rvs(1) < ap)
        if apbool:                                                                      # accept, update self.model
            logger.info(f'Smart-merge move accepted with probability {ap}, p={p}')
            logger.info(f'\t ... merging {idx_cluster1} and {idx_cluster2}')
            # Update model
            self.sIndex = sIndex_prop
            #  ... no new clusters created so we don't need to update cluster counters.
            # Update self
            self._kx1plus, self._kx1plus_indexes = self.model.kx1plus
            self._kx2plus, self._kx2plus_indexes = self.model.kx2plus
        else:                                                                           # reject, do nothing
            logger.info(f'Smart-merge move rejected with probability {ap}, p={p}')

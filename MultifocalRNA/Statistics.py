"""
Implements two types of statistical tests when the seeding lesion is known.
The first test uses a Poisson-binomial distribution and uses as test statistic the number of times a seeding lesion
was correctly indicated.
The second test uses a rank-based approach and tests the rank each lesion has within a patient. The rankscore is the sum
over all ranks of the seeding lesions.
"""

import numpy as np
from scipy.stats import hypergeom
import pandas as pd


def poisson_test_seeding_lesion(test_stat, patient_ids, seeding_lesions, alternative="both"):
    """
    Uses a poisson binomial test to calculate if the test statistics are different in the seeding lesions compared
    to all other lesions in the same patient. A test statistic could be the expression of a gene or the activity of
    a signature. For each statistic we count in how many patient the statistic is the highest in the seeding lesion and
    calculate the significance level using the Poisson-binomial distribution.
    :param test_stat: a pandas DF (samples x test_stats)
    :param patient_ids: The matching patient IDs for the test_stat DF.
    :param seeding_lesions: The seeding lesion, or other lesions of interest.
    :param alternative: Whether to perform a left-, right- or two-sided test
    :return: a pandas DF containing the number of patient where the seeding lesion was correctly identified by the test
    statistic and the associated p-value.
    """
    patient_ids = np.asarray(patient_ids)
    uniq_pat_ids = np.unique(patient_ids)
    probs_per_patient = {}
    n_correct_votes, n_patients = pd.Series(0, index=test_stat.columns), 0

    for pat in uniq_pat_ids:

        pat_mat = test_stat.loc[patient_ids == pat]
        seeding_mask = pat_mat.index.isin(seeding_lesions).astype(int)
        n_seeding = np.sum(seeding_mask)

        if n_seeding == 1:
            binary_mat = pat_mat.values
            binary_mat = binary_mat == np.max(binary_mat, axis=0, keepdims=True)

            n_top = binary_mat.sum(axis=0)

            [M, n, N] = [pat_mat.shape[0], n_seeding, n_top]
            rv = hypergeom(M, n, N)

            probs_per_patient[pat] = 1. - rv.pmf(0)
            n_correct_votes += np.matmul(binary_mat.T, seeding_mask)
            n_patients += 1

        elif n_seeding > 1:
            raise NotImplementedError("The current implementation does not support more "
                                      "than 1 seeding lesion per patient.")

    probs_per_patient = pd.DataFrame(probs_per_patient, index=test_stat.columns)
    odict = {"correct_votes": [], "p-value": []}

    for T_stats in test_stat.columns:
        probs = probs_per_patient.loc[T_stats].values
        pmf = poisson_binom_pmf(probs)
        observed_corr_votes = n_correct_votes.loc[T_stats]

        if alternative.lower() == "both":
            pval = 2 * np.minimum(pmf[observed_corr_votes:].sum(), pmf[:(observed_corr_votes + 1)].sum())
            pval = np.minimum(pval, 1.)

        elif alternative.lower() == "greater":
            pval = pmf[observed_corr_votes:].sum()

        elif alternative.lower() == "less":
            pval = pmf[:(observed_corr_votes + 1)].sum()

        else:
            raise ValueError("The <alternative> argument is not well understood."
                             "Please provide <both> for two-sided testing, <greater> for right-tailed testing "
                             "and <less> for left-tailed testing. "
                             "Note that left tailed testing might require many samples.")

        odict["correct_votes"].append(observed_corr_votes)
        odict["p-value"].append(pval)

    return pd.DataFrame(odict, index=test_stat.columns)


def poisson_binom_pmf(probs):
    """
    Generates a pmf for a list of events with known probabilities
    :param probs: the probabilities of each event
    :return:
    """
    pmf = np.zeros(len(probs) + 1)

    pmf[0] = 1. - probs[0]
    pmf[1] = probs[0]

    for prob in probs[1:]:
        pmf = ((1-prob) * pmf + np.roll(pmf, 1) * prob)

    return pmf


def rankbased_test_seeding_lesion(test_stat, patient_ids, seeding_lesions, alternative="both"):
    """
    Uses a rankbased test to calculate if the test statistics are different in the seeding lesions compared
    to all other lesions in the same patient. A test statistic could be the expression of a gene or the activity of
    a signature. For each statistic we count in how many patient the statistic is the highest in the seeding lesion and
    calculate the significance level using the Poisson-binomial distribution.
    :param test_stat: a pandas DF (samples x test_stats)
    :param patient_ids: The matching patient IDs for the test_stat DF.
    :param seeding_lesions: The seeding lesion, or other lesions of interest.
    :param alternative: Whether to perform a left-, right- or two-sided test
    :return: a pandas DF containing the number of patient where the seeding lesion was correctly identified by the test
    statistic and the associated p-value.
    """
    patient_ids = np.asarray(patient_ids)
    uniq_pat_ids = np.unique(patient_ids)
    rankscore, n_patients = pd.Series(0, index=test_stat.columns), 0
    ranks_dict = {T_stat: [] for T_stat in test_stat.columns}

    for pat in uniq_pat_ids:

        rank_mat = test_stat.loc[patient_ids == pat].rank(axis=0)
        seeding_mask = rank_mat.index.isin(seeding_lesions).astype(int)
        n_seeding = seeding_mask.sum()

        if n_seeding == 1:
            rankscore += np.matmul(rank_mat.T.values, seeding_mask)
            n_patients += 1

            for T_stat in test_stat.columns:
                ranks_dict[T_stat].append(rank_mat[T_stat].to_list())

        elif n_seeding > 1:
            raise NotImplementedError("The current implementation does not support more "
                                      "than 1 seeding lesion per patient. Patient %s has more than 1 seeding lesion."
                                      % pat)

    odict = {"Rankscore": [], "p-value": []}

    for T_stat in test_stat.columns:
        values, pmf = get_background_distribution(ranks_dict[T_stat])
        observed_rank_score = rankscore.loc[T_stat]

        if alternative.lower() == "both":
            upper_mask = values >= observed_rank_score
            lower_mask = values <= observed_rank_score

            pval = 2 * np.minimum(pmf[upper_mask].sum(),
                                  pmf[lower_mask].sum())
            pval = np.minimum(pval, 1.)

        elif alternative.lower() == "greater":
            upper_mask = values >= observed_rank_score
            pval = pmf[upper_mask].sum()

        elif alternative.lower() == "less":
            lower_mask = values <= observed_rank_score
            pval = pmf[lower_mask].sum()

        else:
            raise ValueError("The <alternative> argument is not well understood."
                             "Please provide <both> for two-sided testing, <greater> for right-tailed testing "
                             "and <less> for left-tailed testing. "
                             "Note that left tailed testing might require many samples.")

        odict["Rankscore"].append(observed_rank_score)
        odict["p-value"].append(pval)

    return pd.DataFrame(odict, index=test_stat.columns)


def get_background_distribution(ranks):
    """
    Get the pmf and associated rank scores from a list of rank lists.
    :param ranks: a list containing the ranks per patient. Each entry in the list is a new list, containing the rank of
    each lesion from that patient.
    :return:
    """
    uranks, probs = get_probs_from_ranks(ranks)

    values, pmf = uranks[0], probs[0]

    for i in range(1, len(probs)):
        urank_pat, prob_start_pat = uranks[i], probs[i]

        values, pmf = combine_ranks_and_probs(values, urank_pat, pmf, prob_start_pat)

    return values, pmf


def get_probs_from_ranks(ranks):
    """
    Convert the ranks to probabilities, taking into account ties.
    :param ranks: an iterable containing the ranks per patient
    :return: the unique values of the ranks and the corresponding probabilities
    """
    probs, values = [], []

    for rank_arr in ranks:
        rnks, counts = np.unique(rank_arr, return_counts=True)

        values.append(rnks)
        probs.append(counts/len(rank_arr))

    return values, probs


def groupby_numpy(value_arr, group_arr, aggr_func=np.sum):
    """
    A fast numpy implementation to perform a pandas groupby
    :param value_arr: the array that need to be aggregated
    :param group_arr: the grouping array, all elements of value_arr of the same group are aggregated according to aggr_func
    :param aggr_func: An aggregation function, takes a np array as input and returns a scalar.
    :return: the uniq groups and their aggregated values
    """

    value_arr = np.asarray(value_arr)
    group_arr = np.asarray(group_arr)

    uniq_groups = np.unique(group_arr)
    group2int = {g: i for i, g in enumerate(uniq_groups)}
    group_arr = np.array([group2int[g] for g in group_arr])

    sorted_args = np.argsort(group_arr)
    sorted_groups = group_arr[sorted_args]
    sorted_values = value_arr[sorted_args]

    d = np.diff(np.concatenate([[-np.inf], sorted_groups, [np.inf]]))
    inds = np.where(d != 0)[0]

    aggr_values = np.array([aggr_func(sorted_values[inds[ix]:inds[ix + 1]]) for ix, _
                            in enumerate(uniq_groups)])

    return uniq_groups, aggr_values


def combine_ranks_and_probs(v1, v2, probs1, probs2):
    """
    Function to combine two arrays v1 and v2 into a new array {v = v1 + v2, v1 \in V1 and v2 \in V2}
    while combining the pmfs of v1 and v2 accordingly.
    :param v1:
    :param v2:
    :param probs1:
    :param probs2:
    :return:
    """

    v1, v2 = np.asarray(v1), np.asarray(v2)
    probs1, probs2 = np.asarray(probs1), np.asarray(probs2)

    vs = v1[..., None] + v2[None, ...]
    probs = probs1[..., None] * probs2[None, ...]

    vs = vs.flatten()
    probs = probs.flatten()

    vs, probs = groupby_numpy(probs, vs)

    return vs, probs

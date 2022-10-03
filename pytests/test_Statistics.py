from MultifocalRNA.Statistics import groupby_numpy, get_probs_from_ranks, get_background_distribution, \
    combine_ranks_and_probs, poisson_test_seeding_lesion, rankbased_test_seeding_lesion
import numpy as np


def test_poisson_test_seeding_lesion(example_input_primaries, extended_example_input_primaries):
    # test the shape of the output using a small example
    exp_df, patient_ids, sample_types = example_input_primaries

    pval_df = poisson_test_seeding_lesion(exp_df, patient_ids, seeding_lesions=[0, 1])

    assert 8 == pval_df.shape[0]
    assert 2 == pval_df.shape[1]

    # the last gene denotes a perfect identifier:
    expected_pval = 1/4 * 1/2

    assert np.abs(2 * expected_pval - pval_df.loc["G7"]["p-value"]) < 1e-5
    assert 2 == pval_df.loc["G7"]["correct_votes"]

    # test statistics that are the same for each lesion should get a p-value of 1:
    assert np.abs(1. - pval_df.loc["G5"]["p-value"]) < 1e-5

    # test the exact values of the test using a more complex example:
    exp_df, patient_ids, sample_types, seeds = extended_example_input_primaries

    corr_behaviour = False

    try:
        # this should not work (yet)
        _ = poisson_test_seeding_lesion(exp_df, patient_ids, seeding_lesions=[0, 1])

    except NotImplementedError:
        corr_behaviour = True

    assert corr_behaviour

    # For the extended example we get:
    pval_df = poisson_test_seeding_lesion(exp_df, patient_ids, seeding_lesions=seeds)

    assert 8 == pval_df.shape[0]
    assert 2 == pval_df.shape[1]

    # the last gene denotes a perfect identifier:
    expected_pval = 1/6 ** 5

    assert np.abs(2 * expected_pval - pval_df.loc["G7"]["p-value"]) < 1e-5
    assert 5 == pval_df.loc["G7"]["correct_votes"]

    # test statistics that are the same for each lesion should get a p-value of 1:
    assert np.abs(1. - pval_df.loc["G5"]["p-value"]) < 1e-5


def test_rankbased_test_seeding_lesion(example_input_primaries, extended_example_input_primaries):

    # test the shape of the output using a small example
    exp_df, patient_ids, sample_types = example_input_primaries

    pval_df = rankbased_test_seeding_lesion(exp_df, patient_ids, seeding_lesions=[0, 1])

    assert 8 == pval_df.shape[0]
    assert 2 == pval_df.shape[1]

    # the last gene denotes a perfect identifier:
    expected_pval = 1/4 * 1/2

    assert np.abs(2 * expected_pval - pval_df.loc["G7"]["p-value"]) < 1e-5
    assert 6 == pval_df.loc["G7"]["Rankscore"]

    # test statistics that are the same for each lesion should get a p-value of 1:
    assert np.abs(1. - pval_df.loc["G5"]["p-value"]) < 1e-5

    # test the exact values of the test using a more complex example:
    exp_df, patient_ids, sample_types, seeds = extended_example_input_primaries

    corr_behaviour = False

    try:
        # this should not work (yet)
        _ = rankbased_test_seeding_lesion(exp_df, patient_ids, seeding_lesions=[0, 1])

    except NotImplementedError:
        corr_behaviour = True

    assert corr_behaviour

    # For the extended example we get:
    pval_df = rankbased_test_seeding_lesion(exp_df, patient_ids, seeding_lesions=seeds)

    assert 8 == pval_df.shape[0]
    assert 2 == pval_df.shape[1]

    # the last gene denotes a perfect identifier:
    expected_pval = 1/6 ** 5

    assert np.abs(2 * expected_pval - pval_df.loc["G7"]["p-value"]) < 1e-5
    assert 30 == pval_df.loc["G7"]["Rankscore"]

    # test statistics that are the same for each lesion should get a p-value of 1:
    assert np.abs(1. - pval_df.loc["G5"]["p-value"]) < 1e-5


def test_groupby_numpy():

    test_ids = ["ID1"] * 3 + ["ID2"] * 2 + ["ID3"] * 4
    test_vals = [1] * 3 + [2] * 2 + [3] * 4

    output_ids, output_vals = groupby_numpy(value_arr=test_vals, group_arr=test_ids)
    assert np.all(output_ids == np.array(["ID1", "ID2", "ID3"]))
    assert np.all(output_vals == np.array([3, 4, 12]))

    test_ids = ["ID1"] * 3 + ["ID2"] * 2 + ["ID3"] * 4
    test_vals = [1] * 3 + [2] * 2 + [3] * 4

    output_ids, output_vals = groupby_numpy(value_arr=test_vals, group_arr=test_ids, aggr_func=np.mean)
    assert np.all(output_ids == np.array(["ID1", "ID2", "ID3"]))
    assert np.all(output_vals == np.array([1, 2, 3]))

    test_ids = ["ID1", "ID2", "ID3", "ID2", "ID3", "ID1", "ID2"]
    test_vals = [1, 2, 3, 2, 3, 1, 2]

    output_ids, output_vals = groupby_numpy(value_arr=test_vals, group_arr=test_ids, aggr_func=np.mean)
    assert np.all(output_ids == np.array(["ID1", "ID2", "ID3"]))
    assert np.all(output_vals == np.array([1, 2, 3]))

    test_ids = ["ID1", "ID2", "ID3", "ID2", "ID3", "ID1", "ID2"]
    test_vals = [1, 2, 3, 2, 3, 1, 2]

    output_ids, output_vals = groupby_numpy(value_arr=test_vals, group_arr=test_ids, aggr_func=len)
    assert np.all(output_ids == np.array(["ID1", "ID2", "ID3"]))
    assert np.all(output_vals == np.array([2, 3, 2]))


def test_get_probs_from_ranks():
    test_ranks = [[1, 2, 3, 4, 5], [1, 2, 2, 2, 5], [4, 2.5, 2.5, 1]]

    uranks, probs = get_probs_from_ranks(test_ranks)

    print(uranks)
    print(probs)

    assert np.all(uranks[0] == np.array([1, 2, 3, 4, 5]))
    assert np.all(uranks[1] == np.array([1, 2, 5]))
    assert np.all(uranks[2] == np.array([1, 2.5, 4]))

    assert np.all(probs[0] == np.array([0.2] * 5))
    assert np.all(probs[1] == np.array([0.2, 0.6, 0.2]))
    assert np.all(probs[2] == np.array([0.25, 0.5, 0.25]))


def test_combine_ranks_and_probs():
    v_test1 = [1, 2, 3, 4, 5]
    v_test2 = [1, 2, 3, 4, 5]
    probs1 = [0.2] * 5
    probs2 = [0.2] * 5

    vcomb, probs_comb = combine_ranks_and_probs(v_test1, v_test2, probs1, probs2)

    print(vcomb)
    print(probs_comb)

    assert np.abs(probs_comb.sum() - 1.) < 1e-15


def test_get_background_distribution():
    test_ranks = [[1, 2, 3, 4, 5], [1, 2, 2, 2, 5], [4, 2.5, 2.5, 1]]

    values, pmf = get_background_distribution(test_ranks)
    print(values)
    print(pmf)

    assert np.abs(pmf.sum() - 1.) < 1e-15
    assert np.abs(pmf[0] - .01) < 1e-15
    assert np.abs(pmf[1] - .04) < 1e-15
    assert np.abs(pmf[-1] - .01) < 1e-15
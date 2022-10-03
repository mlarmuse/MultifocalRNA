import pandas as pd
import numpy as np
from MultifocalRNA.MultifocalDataset import MultifocalDataset
from MultifocalRNA.CentroidClassifier import spearman_dist, all_matrix_distances, pearson_dist


def test_sort_by_patient_id_and_lesion_type():

    exp_df = pd.DataFrame(np.random.randn(5, 8),
                          index=[i for i in range(5)],
                          columns=["G%i" % i for i in range(8)])

    patient_ids = [1, 2, 2, 1, 1]
    sample_types = ["primary", "meta", "primary", "meta", "an"]

    mf = MultifocalDataset(exp_df, patient_ids=patient_ids, sample_types=sample_types)

    assert np.all(mf.df.index.values == np.array([4, 3, 0, 1, 2]))
    assert np.all(mf.patient_ids == np.array([1, 1, 1, 2, 2]).astype(str))
    assert np.all(mf.sample_types == np.array(["an", "meta", "primary", "meta", "primary"]))


def test_create_custom_sample_ids():

    exp_df = pd.DataFrame(np.random.randn(6, 8),
                          index=[i for i in range(6)],
                          columns=["G%i" % i for i in range(8)])

    patient_ids = [1, 2, 2, 1, 1, 1]
    sample_types = ["primary", "meta", "primary", "meta", "an", "primary"]
    mf = MultifocalDataset(exp_df, patient_ids=patient_ids, sample_types=sample_types)

    sids = mf.create_custom_sample_ids()

    assert np.all(sids == np.array(['1_an1', '1_meta1', '1_primary1', '1_primary2', '2_meta1', '2_primary1']))


def test_calculate_centroids(example_input_small, badly_formatted_signature_dict):
    exp_df, patient_ids, sample_types = example_input_small

    mf = MultifocalDataset(exp_df, patient_ids, sample_types)

    centroids_per_signature = mf.calculate_centroids(badly_formatted_signature_dict, min_sign_size=3)

    correct_centroids = exp_df.groupby(sample_types).mean()

    # assert centroids_per_signature["Signature 1"].centroids == centroids[["G1", "G3", "G5", "G7"]]
    print(centroids_per_signature["Signature 2"].centroids)
    print(correct_centroids[["G0", "G2", "G4", "G6"]])
    assert np.all(np.abs(centroids_per_signature["Signature 2"].centroids -
                         correct_centroids[["G0", "G2", "G4", "G6"]]) < 1e-5)
    assert np.all(np.abs(centroids_per_signature["Signature 1"].centroids -
                         correct_centroids[["G1", "G3", "G5", "G7"]]) < 1e-5)

    centroids_per_signature = mf.calculate_centroids(badly_formatted_signature_dict, min_sign_size=5)

    print(centroids_per_signature)

    assert len(centroids_per_signature) == 0


def test_check_signatures(example_input_small, badly_formatted_signature_dict):
    exp_df, patient_ids, sample_types = example_input_small

    mf = MultifocalDataset(exp_df, patient_ids, sample_types)

    corr_dict = mf.check_signatures(badly_formatted_signature_dict)

    assert np.all(np.asarray(corr_dict["Signature 1"]) == np.array(['G1', 'G3', 'G5', 'G7']))
    assert np.all(np.asarray(corr_dict["Signature 2"]) == np.array(['G0', 'G2', 'G4', 'G6']))


def test_calculate_centroid_probabilities(example_input_small, example_signature_dict):
    exp_df, patient_ids, sample_types = example_input_small

    mf = MultifocalDataset(exp_df, patient_ids, sample_types)

    prob_df = mf.calculate_centroid_probabilities(sample_type="meta",
                                                  signature_dict=example_signature_dict,
                                                  min_sign_size=3)

    print(prob_df)

    correct_centroids = exp_df.groupby(sample_types).mean()

    prob_dict = {}

    for sign, genes in example_signature_dict.items():
        affinities = 1. - spearman_dist(exp_df[genes].values, correct_centroids[genes].values)
        affinities2 = 1. - all_matrix_distances(exp_df[genes].values, correct_centroids[genes].values)
        affinities = affinities/np.sum(affinities, axis=0, keepdims=True)

        prob_dict[sign] = affinities[:, correct_centroids.index.values == "meta"].flatten()

    correct_prob_df = pd.DataFrame(prob_dict)

    print(correct_prob_df)
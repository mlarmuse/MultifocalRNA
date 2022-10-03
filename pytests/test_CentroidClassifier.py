import numpy as np
from MultifocalRNA.CentroidClassifier import normalize_distances, spearman_mat, pearson_mat, CentroidClassifier, \
    spearman_dist, pearson_dist, all_matrix_distances, pairwise_distances, spearman_dist
import time
from scipy.stats import spearmanr, pearsonr


def test_normalize_distances():
    test_data1 = np.array([[0.4, 0.7, 0.5],
                           [0.4, 0.4, 0.4],
                           [0, 0, 0]])
    norm_distances = normalize_distances(test_data1)

    assert np.abs(norm_distances[2, 0] - 1./3) < 1e-5
    assert np.all(np.abs(np.ones(3) - norm_distances.sum(axis=1)) < 1e-5)


def test_corrmat():
    testmat1 = np.array([[1, 2, 3, 4],
                         [4, 5, 6, 7],
                         [8, 5, 3, 5]])

    testmat2 = np.array([[7, 2, 3, 4],
                         [3, 5, 9, 7],
                         [1, 5, 10, 5]])

    corrmat = np.zeros((testmat1.shape[0], testmat2.shape[0]))

    for i, v1 in enumerate(testmat1):
        for j, v2 in enumerate(testmat2):
            corrmat[i, j] = spearmanr(v1, v2)[0]

    corrmat2 = spearman_mat(testmat1, testmat2)
    assert np.all(np.abs(corrmat - corrmat2) < 1e-5)

    corr_vec = spearman_mat(testmat1.flatten(), testmat2.flatten())
    rho = spearmanr(testmat1.flatten(), testmat2.flatten())[0]
    assert np.all(np.abs(rho - corr_vec) < 1e-5)

    test_vec = np.array([1, 2, 3, 4])

    test = spearman_mat(testmat1, test_vec)
    assert np.abs(1. - test[0]) < 1e-5
    assert np.abs(1. - test[1]) < 1e-5


def test_corrmat_pearson():
    testmat1 = np.array([[1, 2, 3, 4],
                         [4, 5, 6, 7],
                         [8, 5, 3, 5]])

    testmat2 = np.array([[7, 2, 3, 4],
                         [3, 5, 9, 7],
                         [1, 5, 10, 5]])

    corrmat = np.zeros((testmat1.shape[0], testmat2.shape[0]))

    for i, v1 in enumerate(testmat1):
        for j, v2 in enumerate(testmat2):
            corrmat[i, j] = pearsonr(v1, v2)[0]

    corrmat2 = pearson_mat(testmat1, testmat2)
    assert np.all(np.abs(corrmat - corrmat2) < 1e-5)

    corr_vec = pearson_mat(testmat1.flatten(), testmat2.flatten())
    rho = pearsonr(testmat1.flatten(), testmat2.flatten())[0]
    assert np.all(np.abs(rho - corr_vec) < 1e-5)

    test_vec = np.array([1, 2, 3, 4])

    test = pearson_mat(testmat1, test_vec)
    assert np.abs(1. - test[0]) < 1e-5
    assert np.abs(1. - test[1]) < 1e-5


def test_predict_proba():
    X_train = np.array([[1, 2, 3, 4],
                         [4, 5, 6, 7],
                         [8, 5, 3, 5]])

    y_train = np.array([0, 1, 1])

    X_test = np.array([[7, 2, 3, 4],
                       [3, 5, 9, 7],
                       [1, 5, 10, 5]])

    clf = CentroidClassifier()
    clf.fit(X_train, y_train)

    preds1 = clf.predict_proba(X_test)

    correct_centroids = np.vstack((X_train[y_train == 0].mean(axis=0), X_train[y_train == 1].mean(axis=0)))

    corr_affinities = 1. - spearman_dist(correct_centroids, X_test)
    norm_constant = np.sum(corr_affinities, axis=0, keepdims=True)
    probs = corr_affinities/norm_constant

    assert np.all(np.abs(probs.T - preds1) < 1e-5)

    preds2 = clf.predict_proba(X_test, metric="euclidean")

    corr_affinities = 1. - pairwise_distances(correct_centroids, X_test, metric="euclidean")
    norm_constant = np.sum(corr_affinities, axis=0, keepdims=True)
    probs2 = corr_affinities / norm_constant

    assert np.all(np.abs(probs2.T - preds2) < 1e-5)


def test_all_matrix_distances(example_input_small, example_signature_dict):
    exp_df, patient_ids, sample_types = example_input_small
    correct_centroids = exp_df.groupby(sample_types).mean()

    for sign, genes in example_signature_dict.items():
        affinities = 1. - spearman_dist(exp_df[genes].values, correct_centroids[genes].values)
        affinities2 = 1. - all_matrix_distances(exp_df[genes].values, correct_centroids[genes].values)

        assert np.all(np.abs(affinities - affinities2) < 1e-5)

    for sign, genes in example_signature_dict.items():
        affinities = 1. - pearson_dist(exp_df[genes].values, correct_centroids[genes].values)
        affinities2 = 1. - all_matrix_distances(exp_df[genes].values, correct_centroids[genes].values,
                                                metric="pearson")

        assert np.all(np.abs(affinities - affinities2) < 1e-5)

    for sign, genes in example_signature_dict.items():
        affinities = 1. - pairwise_distances(exp_df[genes].values, correct_centroids[genes].values)
        affinities2 = 1. - all_matrix_distances(exp_df[genes].values, correct_centroids[genes].values,
                                                metric="euclidean")

        assert np.all(np.abs(affinities - affinities2) < 1e-5)



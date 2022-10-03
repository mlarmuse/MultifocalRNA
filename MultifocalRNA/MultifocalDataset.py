import numpy as np
import pandas as pd
from MultifocalRNA.CentroidClassifier import CentroidClassifier, all_matrix_distances
import copy
from MultifocalRNA.Statistics import poisson_test_seeding_lesion


class MultifocalDataset:

    def __init__(self, exp_df, patient_ids, sample_types=None):
        """
        :param exp_df: a n_samples x n_genes pandas DF containing unique sample ids as indices and unique gene ids as rows
        :param patient_ids: The corresponding patient ids for the sample_ids in the index of exp_df.
        For multifocal studies, different sample_ids should stem from the same patient id
        :param sample_types: the type of each sample. This is necessary if one wants to e.g. compare between samples.
        :return:
        """

        self.df = exp_df
        self.patient_ids = np.asarray(patient_ids).astype(str)
        self.sample_types = np.asarray(sample_types).astype(str)

        self.uniq_patients = np.unique(self.patient_ids)
        self.n_patients = len(self.uniq_patients)
        self.centroids = None
        self.seeding_lesions = None

        print("Initialized a dataset with %i samples and %i genes for %i patients" % (*self.df.shape,
                                                                                      self.n_patients))

        self.sort_by_patient_id_and_lesion_type()

    @property
    def genes(self):
        return self.df.columns.values.astype(str)

    def check_signatures(self, signature_dict):
        """
        Helper function to filter the user provided signature dict.
        :param signature_dict:
        :return:
        """
        signature_dict = copy.deepcopy(signature_dict)

        dataset_genes = self.genes

        return {sign: np.intersect1d(dataset_genes, sign_genes) for sign, sign_genes in signature_dict.items()}

    def sort_by_patient_id_and_lesion_type(self):
        if self.sample_types is not None:
            idxs = np.lexsort((self.sample_types, self.patient_ids))

        else:
            idxs = np.argsort(self.patient_ids)

        self.sort_all(idxs)

    def sort_all(self, idx):

        self.df = self.df.iloc[idx]
        self.patient_ids = self.patient_ids[idx]
        self.sample_types = self.sample_types[idx]

    def create_custom_sample_ids(self, sep="_"):
        """
        Create custom sample_ids of the form <patient_id><sep><sample_type><int>, where the int is incremental and starts at 1.
        :param sep:
        :return:
        """
        self.id_sep = sep

        curr_pat = self.patient_ids[0]
        curr_tissue_type = self.sample_types[0]
        curr_counter = 1
        new_sample_ids = []

        for pat, sample_type in zip(self.patient_ids, self.sample_types):

            if pat != curr_pat:
                curr_counter = 1
                curr_pat = pat

            if sample_type != curr_tissue_type:
                curr_counter = 1
                curr_tissue_type = sample_type

            new_sample_ids.append(curr_pat + sep + curr_tissue_type + str(curr_counter))
            curr_counter += 1

        new_sample_ids = np.asarray(new_sample_ids)

        return new_sample_ids

    def calculate_centroids(self, signature_dict, min_sign_size=5):
        """
        Calculate the centroid for signatures provided in signature dict.
        For each signature there will be one centroid per tissue type.
        :param signature_dict:
        :param min_sign_size:  Only consider signatures with more than this many genes.
        :return:
        """
        centroids_per_signature = calculate_centroids(self.df, self.sample_types, signature_dict,
                                                      min_sign_size=min_sign_size)
        self.centroids = centroids_per_signature
        return centroids_per_signature

    def calculate_centroid_probabilities(self, sample_type, signature_dict, min_sign_size=5,
                                               recalculate_centroids=False, metric="spearman"):
        """
        Calculates the distance to the centroid for each of the signatures provided in signature dict.
        If the centroids already were calculated before, those will be used. To override this use
        recalculate_centroids=True.
        :param sample_type:
        :param signature_dict:
        :param min_sign_size:
        :param: recalculate_centroids:
        :return:
        """

        signature_dict = self.check_signatures(signature_dict)

        if self.centroids is None and not recalculate_centroids:
            self.calculate_centroids(signature_dict=signature_dict, min_sign_size=min_sign_size)

        dists_to_centroid = {}

        for sign, centroid_df in self.centroids.items():
            signature_genes = signature_dict[sign]

            if len(signature_genes) > min_sign_size:
                clf = self.centroids[sign]
                dists_to_centroid[sign] = clf.predict_proba(self.df[signature_dict[sign]].values,
                                                            metric=metric)[:, clf.classes == sample_type].flatten()

        return pd.DataFrame(dists_to_centroid, index=self.df.index)

    def perform_voting(self, probs_per_signature=None, **calc_prob_kwargs):
        """
        Identify the seeding lesion using resemblance to metastatic samples.
        :param signature_dict: A signature dict to be provided, if signature dict is None,
        individual genes are used to do the voting.
        :return:
        """

        if probs_per_signature is None:
            self.calculate_centroid_probabilities(**calc_prob_kwargs)

        patients = np.array([s.split("_")[1] for s in probs_per_signature.index.values])
        binary_dfs, votes_df, seeds, vote_fractions = [], [], [], []
        non_seeds = []
        pat_index = []

        for pid in self.uniq_patients:
            print(pid)
            mat = probs_per_signature.loc[patients == pid]

            norm_votes = mat.astype(int) / np.sum(mat.values, axis=0, keepdims=True)
            norm_votes = norm_votes.fillna(0)
            votes_df.append(norm_votes)

            votes = norm_votes.sum(axis=1)
            pat_seeds = list(votes.index.values[votes.values == votes.max()])
            seeds += pat_seeds
            non_seeds += list(votes.index.values[votes.values == votes.min()])

            pat_index.append(pid)

            if len(pat_seeds) == 0:
                print("No seed found for patient %s" % pid)

        votes_df = pd.concat(votes_df, axis=1, sort=True)
        self.seeding_lesions = seeds

        return votes_df, seeds, non_seeds

    def get_signature_matrix(self, sample_type):
        """
        Get all centroids as a DF (n_signatures x
        :param sample_type:
        :return:
        """
        return pd.concat([df[sample_type] for sign, df in self.centroids.values()], axis=1)

    def perform_significance_testing(self):

        if self.seeding_lesions is None:
            pass
            # TODO


def get_all_genes_from_dict(signature_dict):
    return np.unique([g for l in signature_dict.values for g in l])


def calculate_centroids(exp, y, signature_dict, min_sign_size=5):
    """
    Calculates the CentroidClassifier object for each signature
    :param exp:
    :param y:
    :param signature_dict:
    :param min_sign_size:
    :return:
    """
    centroids_per_signature = {}

    for signature, sign_genes in signature_dict.items():

        common_genes = np.intersect1d(sign_genes, exp.columns.values)

        if len(common_genes) > min_sign_size:
            X = exp[common_genes]

            clf = CentroidClassifier()
            clf.fit(X, y)

            centroids_per_signature[signature] = clf

    return centroids_per_signature


def get_random_signatures(signature_size, N_sets, background_genes="all"):
    pass


def gene_based_voting():
    pass


def signature_based_voting():
    pass


def get_signatures_size(signatures_dict):
    return pd.Series({signature: len(np.unique(values)) for signature, values in signatures_dict.items()})


def get_unqiue_signature_sizes(signatures_dict):
    return pd.unique(get_signatures_size(signatures_dict))


def perform_significance_testing(test_stat_df, patient_ids, seeding_lesions, test_type="rankbased"):
    """
    Perform statistical testing when the seeding lesion is known.

    :param test_stat_df:
    :param patient_ids:
    :param seeding_lesions:
    :param test_type:
    :return:
    """












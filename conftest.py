import pandas as pd
import numpy as np
import pytest


@pytest.fixture(scope='module')
def example_input_small():

    exp_df = pd.DataFrame(np.random.randn(6, 8),
                          index=[i for i in range(6)],
                          columns=["G%i" % i for i in range(8)])

    patient_ids = [1, 2, 2, 1, 1, 1]
    sample_types = ["primary", "meta", "primary", "meta", "an", "primary"]
    return exp_df, patient_ids, sample_types


@pytest.fixture(scope='module')
def example_input_primaries():

    data_mat = [[2, 0, 1, 1, 1, 1, 1, 6],
                [2, 0, 1, 1, 1, 1, 1, 5],
                [1, 1, 1, 1, 1, 1, 1, 4],
                [1, 1, 1, 1, 1, 1, 1, 3],
                [1, 1, 1, 1, 1, 1, 1, 2],
                [1, 1, 1, 1, 1, 1, 1, 1]]
    exp_df = pd.DataFrame(data_mat,
                          index=[i for i in range(6)],
                          columns=["G%i" % i for i in range(8)])

    patient_ids = [1, 2, 2, 1, 1, 1]
    sample_types = ["primary"] * 6
    return exp_df, patient_ids, sample_types


@pytest.fixture(scope='module')
def extended_example_input_primaries():

    data_mat = [[2, 0, 1, 1, 1, 1, 1, 6],
                [2, 0, 1, 1, 1, 1, 1, 5],
                [1, 1, 1, 1, 1, 1, 1, 4],
                [1, 1, 1, 1, 1, 1, 1, 3],
                [1, 1, 1, 1, 1, 1, 1, 2],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [2, 0, 1, 1, 1, 1, 1, 6],
                [2, 0, 1, 1, 1, 1, 1, 5],
                [1, 1, 1, 1, 1, 1, 1, 4],
                [1, 1, 1, 1, 1, 1, 1, 3],
                [1, 1, 1, 1, 1, 1, 1, 2],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [2, 0, 1, 1, 1, 1, 1, 6],
                [2, 0, 1, 1, 1, 1, 1, 5],
                [1, 1, 1, 1, 1, 1, 1, 4],
                [1, 1, 1, 1, 1, 1, 1, 3],
                [1, 1, 1, 1, 1, 1, 1, 2],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [2, 0, 1, 1, 1, 1, 1, 6],
                [2, 0, 1, 1, 1, 1, 1, 5],
                [1, 1, 1, 1, 1, 1, 1, 4],
                [1, 1, 1, 1, 1, 1, 1, 3],
                [1, 1, 1, 1, 1, 1, 1, 2],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [2, 0, 1, 1, 1, 1, 1, 6],
                [2, 0, 1, 1, 1, 1, 1, 5],
                [1, 1, 1, 1, 1, 1, 1, 4],
                [1, 1, 1, 1, 1, 1, 1, 3],
                [1, 1, 1, 1, 1, 1, 1, 2],
                [1, 1, 1, 1, 1, 1, 1, 1]]

    exp_df = pd.DataFrame(data_mat,
                          index=[i for i in range(len(data_mat))],
                          columns=["G%i" % i for i in range(8)])

    patient_ids = [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 6
    sample_types = ["primary"] * 6
    seeding_lesions = [0, 6, 12, 18, 24]

    return exp_df, patient_ids, sample_types, seeding_lesions


@pytest.fixture(scope='module')
def example_signature_dict():

    example = {"Signature 1": ["G1", "G3", "G5", "G7"],
               "Signature 2": ["G2", "G4", "G6", "G0"]}

    return example


@pytest.fixture(scope='module')
def badly_formatted_signature_dict():
    """
    Example of a signature dict that contains duplicates and genes that cannot occur in the experimental data
    :return:
    """
    example = {"Signature 1": ["G1", "G3", "G5", "G7", "G_wrong"],
               "Signature 2": ["G2", "G4", "G6", "G0", "G0", "G0"]}

    return example
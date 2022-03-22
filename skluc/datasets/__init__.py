# -*- coding: utf-8 -*-
"""
This module contains utility functions to download datasets and return them as X,Y (or X, None) numpy ndarray.
"""
import os
import pathlib
import tempfile
from typing import Literal

import numpy as np
import pandas as pd
import re
import zipfile
import tarfile
import matplotlib.pyplot as plt
import cv2

from pathlib import Path
from sklearn import preprocessing
from sklearn.datasets import make_blobs, fetch_kddcup99, fetch_covtype
from skluc.datasets.bio_data_utils import load_expression, clean_labels, load_phenotype, \
    load_methylation, _load_phenotypes,  clean_sample_IDs
from skluc.utils.datautils.imageutils import crop_center
from skluc.utils.osutils import download_file, logger


def load_kddcup04bio():
    data_url = "http://cs.joensuu.fi/sipu/datasets/KDDCUP04Bio.txt"

    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        matfile_path = download_file(data_url, d_tmp)
        data = pd.read_csv(matfile_path, delim_whitespace=True)

    return data.values, None


def load_census1990():
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"

    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        matfile_path = download_file(data_url, d_tmp)
        data = pd.read_csv(matfile_path)

    return data.values[1:], None # remove the `caseId` attribute


def load_plants():
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/plants/plants.data"

    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        file_path = download_file(data_url, d_tmp)

        with open(file_path, 'r', encoding="ISO-8859-15") as f:
            plants = f.readlines()

    set_plants_attributes = set()
    lst_plants = []
    for plant_line in plants:
        plant_line_no_name = [v.strip() for v in plant_line.split(',')[1:]]
        lst_plants.append(plant_line_no_name)
        set_plants_attributes.update(plant_line_no_name)

    arr_plants_attributes = np.array([v for v in set_plants_attributes])
    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    onehot_encoder.fit(arr_plants_attributes.reshape(-1, 1))

    for i, plant_line_no_name in enumerate(lst_plants):
        plant_line_oh = np.sum(onehot_encoder.transform(np.array(plant_line_no_name).reshape(-1, 1)), axis=0)
        lst_plants[i] = plant_line_oh

    arr_lst_plants = np.array(lst_plants)

    return arr_lst_plants, None


def load_caltech(final_size):
    from google_drive_downloader import GoogleDriveDownloader as gdd

    data_url = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"

    lst_images = []
    lst_classes_idx = []

    with tempfile.TemporaryDirectory() as d_tmp:
        destination_path = d_tmp + "/256_ObjectCategories.tar"
        logger.debug(f"Downloading file from url {data_url} to temporary file {destination_path}")
        gdd.download_file_from_google_drive(file_id='1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK',
                                            dest_path=destination_path)
        tarfile_path = Path(destination_path)

        dir_path = Path(d_tmp)

        tf = tarfile.open(tarfile_path)
        tf.extractall(dir_path / "caltech256")
        tf.close()
        for root, dirs, files in os.walk(dir_path / "caltech256"):
            print(root)
            label_class = root.split("/") [-1]
            splitted_label_class = label_class.split(".")
            if splitted_label_class[-1] == "clutter":
                continue
            if len(splitted_label_class) > 1:
                label_idx = int(splitted_label_class[0])
            else:
                continue

            for file in files:
                path_img_file = Path(root) / file
                try:
                    img = plt.imread(path_img_file)
                except:
                    continue
                aspect_ratio = max(final_size / img.shape[0], final_size / img.shape[1])
                new_img = cv2.resize(img, dsize=(0,0), fx=aspect_ratio, fy=aspect_ratio)
                new_img = crop_center(new_img, (final_size, final_size, 3))

                if new_img.shape == (final_size, final_size):
                    new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)


                lst_images.append(new_img.flatten())
                lst_classes_idx.append(label_idx)

        X = np.vstack(lst_images)
        y = np.array(lst_classes_idx)

        print(X.shape)
        print(y.shape)

    return X, y


def load_coil20(final_size):
    data_url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
    regex_file_names = re.compile(r'obj(\d+).+')

    lst_images = []
    lst_classes_idx = []
    with tempfile.TemporaryDirectory() as d_tmp:
        temp_dir = Path(d_tmp)
        zipfile_path = Path(download_file(data_url, d_tmp))
        # zipfile_path = "/home/luc/Téléchargements/coil-20-proc.zip"

        zip_dir = temp_dir / "zip_extraction_dir"
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            zip_ref.extractall(zip_dir)
        for root, dirs, files in os.walk(zip_dir):
            for file_name in files:
                path_file = temp_dir / root / file_name
                img_class = int(regex_file_names.match(file_name).group(1)) - 1
                img_array = plt.imread(str(path_file.absolute()))
                aspect_ratio = max(final_size / img_array.shape[0], final_size / img_array.shape[1])
                new_img_array = cv2.resize(img_array, dsize=(0, 0), fx=aspect_ratio, fy=aspect_ratio)

                lst_images.append(new_img_array.flatten())
                lst_classes_idx.append(img_class)

    X = np.vstack(lst_images)
    y = np.array(lst_classes_idx)

    return X, y


def load_kddcup99():
    X, y = fetch_kddcup99(shuffle=True, return_X_y=True)
    df_X = pd.DataFrame(X)
    X = pd.get_dummies(df_X, columns=[1, 2, 3], prefix=['protocol_type', "service", "flag"]).values.astype(np.float32)
    max_by_col = np.max(X, axis=0)
    min_by_col = np.min(X, axis=0)
    X = (X - min_by_col) / (max_by_col - min_by_col)
    X = X[:,~np.any(np.isnan(X), axis=0)]
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y.reshape(-1, 1))
    return X, y


def load_covtype():
    X, y = fetch_covtype(shuffle=True, return_X_y=True)
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y.reshape(-1, 1))
    return X, y


def generator_blobs_data(data_size, size_batch, nb_features, nb_centers):
    total_nb_chunks = int(data_size // size_batch)
    init_centers = np.random.uniform(-10.0, 10.0, (nb_centers, nb_features))
    for i in range(total_nb_chunks):
        logger.info("Chunk {}/{}".format(i + 1, total_nb_chunks))
        X, y = make_blobs(size_batch, n_features=nb_features, centers=init_centers, cluster_std=12.)
        yield X, y


def load_bio_data(database: Literal["PANCAN", "GDC"], data_kind: Literal["expression", "methylation", "whole"], data_path: pathlib.Path):
    """

    Several databases on UCSC xenabrowser: TCGA Pan-Cancer (PANCAN), GDC TCGA, legacy TCGA.

    Data source
    -----------
    url = {}
    url['PANCAN'] = {
        'phenotype': 'https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp',
        'phenotype_subtypes': 'https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGASubtype.20170308.tsv.gz',
        'expression_counts': 'https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz',
        'methylation450': 'https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz'
    }

    url['GDC'] = {
        'phenotype': 'https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-{}.GDC_phenotype.tsv.gz'.format(cancer),
        'expression_counts': 'https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-{}.htseq_counts.tsv.gz'.format(cancer),
        'ID_gene_mapping': 'https://gdc-hub.s3.us-east-1.amazonaws.com/download/gencode.v22.annotation.gene.probeMap',
        'methylation450': 'https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-{}.methylation450.tsv.gz'.format(cancer)
    }

    url['legacy'] = {
        'phenotype': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.{}.sampleMap%2F{}_clinicalMatrix'.format(cancer, cancer),
        'expression_counts': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.{}.sampleMap%2FHiSeqV2.gz'.format(cancer),
        'ID_gene_mapping': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap%2Fhugo_gencode_good_hg19_V24lift37_probemap'
    }

    Parameters
    ----------
    database
    data_kind
    data_path

    Returns
    -------

    """
    # DATASET 1: classify cancer types from biological data
    ## The dataset contains samples from 33 cancer types.
    # database = 'PANCAN'
    # cancer = None
    # label_name = "cancer type abbreviation"
    # or
    # DATASET 2: classify breast cancer stages from biological data
    ## The data contains several possible labels, not necessarily related to the disease.
    ## Examples: "age_at_initial_pathologic_diagnosis", "menopause_status", "pathologic_M", "pathologic_N", "pathologic_T"
    ## For information, T = tumor size, M = metastasis, N = lymph nodes spreading.
    ## Tumor stages are determined from T,M,N.
    ## More details on https://www.cancer.org/cancer/breast-cancer/understanding-a-breast-cancer-diagnosis/stages-of-breast-cancer.html
    # database = 'GDC'
    # cancer = 'BRCA'
    # label_name = "tumor_stage.diagnoses"
    if database == "PANCAN":
        cancer = None
        label_name = "cancer type abbreviation"
    else:
        cancer = 'BRCA'
        label_name = "tumor_stage.diagnoses"


    # download_dataset(d_tmp, database, cancer)

    labels, label_key, label_map = _load_phenotypes(data_path, database, cancer, label_name)
    if data_kind == "expression" or data_kind == "whole":
        samples_expression = load_expression(data_path, database, cancer)
        sample_ids_expression = clean_sample_IDs(samples_expression, labels, label_key)
        samples_expression = samples_expression.loc[sample_ids_expression]
        samples = samples_expression
        sample_ids = sample_ids_expression
    if data_kind == "methylation" or data_kind == "whole":
        samples_methylation = load_methylation(data_path, database, cancer)
        sample_ids_methylation = clean_sample_IDs(samples_methylation, labels, label_key)
        samples_methylation = samples_methylation.loc[sample_ids_methylation]
        samples = samples_methylation
        sample_ids = sample_ids_methylation

    if data_kind == "whole":
        sample_ids = list(set(sample_ids_expression).intersection(sample_ids_methylation))
        samples_expression = samples_expression.loc[sample_ids]
        samples_methylation = samples_methylation.loc[sample_ids]
        samples = pd.concat([samples_expression, samples_methylation], axis=1)

    try:
        samples_x = samples.values
        labels = labels.loc[sample_ids]
        labels_y = labels.values
        labels_y = np.array([label_map[val] for val in labels_y])
    except NameError:
        raise NameError(f"`data_kind` function attribute should be in Literal['expression', 'methylation', 'whole'] but is {data_kind}")

    return samples_x, labels_y



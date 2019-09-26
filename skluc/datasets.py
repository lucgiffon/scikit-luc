# -*- coding: utf-8 -*-

import os
import tempfile
import numpy as np
import pandas
from pathlib import Path
import tarfile
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from skluc.utils.datautils.imageutils import crop_center
from skluc.utils.osutils import download_file, logger
import cv2


def load_kddcup04bio():
    data_url = "http://cs.joensuu.fi/sipu/datasets/KDDCUP04Bio.txt"

    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        matfile_path = download_file(data_url, d_tmp)
        data = pandas.read_csv(matfile_path, delim_whitespace=True)

    return data.values

def load_census1990():
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"

    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        matfile_path = download_file(data_url, d_tmp)
        data = pandas.read_csv(matfile_path)

    return data.values[1:] # remove the `caseId` attribute

def load_caltech(final_size):
    data_url = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"

    lst_images = []
    lst_classes_idx = []


    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        tarfile_path = Path(download_file(data_url, d_tmp))
        # tarfile_path = Path("/home/luc/Téléchargements/256_ObjectCategories.tar")

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42, stratify=y)

    return (X_train, y_train), (X_test, y_test)

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

    return arr_lst_plants


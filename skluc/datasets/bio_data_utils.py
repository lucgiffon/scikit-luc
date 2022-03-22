"""
Utility code to download data from the PANCAN and GDC databases.

The original author of this code is Bontonou Myriam PhD.
"""
import pathlib
from typing import Optional, NoReturn, Literal, Dict, Union
from six.moves import urllib
import gzip
import os
import pandas as pd
import numpy as np
from loguru import logger


# Labels
def get_possible_classes(database: Literal["PANCAN", "GDC"], cancer: Optional[Literal["BRCA"]]) -> list:
    """
    Return the column with the possible classes in database and cancer type.

    Parameters
    ----------
    database:
        Database where to look for data.
    cancer:
        Kind of cancer.

    Returns
    -------
        List of columns where to look for possible classes.
    """
    possible = {}
    possible['PANCAN-None'] = ["cancer type abbreviation", ]
    possible['GDC-BRCA'] = ["tumor_stage.diagnoses", "age_at_initial_pathologic_diagnosis", "menopause_status",
                            "pathologic_M", "pathologic_N", "pathologic_T"]
    possible_classes = possible[f"{database}-{cancer}"]
    return possible_classes


def get_unwanted_labels(database: Literal["PANCAN", "GDC"], cancer: Optional[Literal["BRCA"]]) -> list:
    """
    Returns the list of unwanted label in the data from `database` and for `cancer` type.

    Parameters
    ----------
    database:
        The database to query.
    cancer:
        The type of cancer.

    Returns
    -------
        The list of unwanted labels.
    """
    unwanted = {}
    unwanted['PANCAN-None'] = []
    unwanted['GDC-BRCA'] = ['not reported', 'nan', 'stage x', 'MX', 'NX', 'TX']
    try:
        unwanted_labels = unwanted[f"{database}-{cancer}"]
    except KeyError:
        unwanted_labels = []
    return unwanted_labels


def clean_labels(label_key: list, database: Literal["PANCAN", "GDC"], cancer: Optional[Literal["BRCA"]]) -> list:
    """
    Remove unwanted labels from the list of base labels.

    Parameters
    ----------
    label_key:
        The list of base labels to clean.
    database:
        The database to query.
    cancer:
        The type of cancer.

    Returns
    -------
        The currated list of labels.
    """
    unwanted_labels = get_unwanted_labels(database, cancer)
    for value in unwanted_labels:
        try:
            label_key.remove(value)
        except ValueError:
            logger.warning(f"Can't remove {value}. It is already absent from base list of labels.")
    # Remove nan numbers
    label_key = [label for label in label_key if label == label]
    return label_key


def load_expression(data_path: Union[str, pathlib.Path], database: Literal["PANCAN", "GDC"], cancer: Optional[Literal["BRCA"]]) -> pd.DataFrame:
    """
    Load from data_path, the Dataframe of gene expression with the given database and cancer type.

    Parameters
    ----------
    data_path:
        The path where to find the "cancer" file, containing the possible values for the cancer argument.
    database:
        The database to query.
    cancer:
        The type of cancer.

    Returns
    -------
        The dataframe of gene expression with patient as rows and methylation sites as columns.
    """
    file_path = os.path.join(data_path, database, '{}_count.tsv.gz'.format(cancer))
    column_name = get_index_column_name(database, cancer, 'expression')
    df = pd.read_csv(file_path, compression="gzip", sep="\t", index_col=column_name)
    df = df.transpose()
    # remove unwanted columns with __ in front
    cols = [c for c in df.columns if c[:2] == '__']
    df = df.drop(labels=cols, axis=1)
    # Remove cols whose value is missing for some samples
    df = df.dropna(axis=1)
    return df


def load_methylation(data_path: Union[str, pathlib.Path], database: Literal["PANCAN", "GDC"], cancer: Optional[Literal["BRCA"]]) -> pd.DataFrame:
    """
    Load from data_path, the Dataframe of methylation sites with the given database and cancer type.

    Parameters
    ----------
    data_path:
        The path where to find the "cancer" file, containing the possible values for the cancer argument.
    database:
        The database to query.
    cancer:
        The type of cancer.

    Returns
    -------
        The dataframe of methylation profiles with patient as rows and methylation sites as columns.
    """
    file_path = os.path.join(data_path, database, '{}_methylation.tsv.gz'.format(cancer))
    column_name = get_index_column_name(database, cancer, 'methylation')
    df = pd.read_csv(file_path, compression="gzip", sep="\t", index_col=column_name)
    df = df.transpose()
    # Remove cols whose value is missing for some samples
    df = df.dropna(axis=1)
    return df


def load_phenotype(data_path: str, database: Literal["PANCAN", "GDC"], cancer: Optional[Literal["BRCA"]]):
    """
    Load from data_path, the Dataframe of phenotypes with the given database and cancer type.

    Parameters
    ----------
    data_path:
        The path where to find the "cancer" file, containing the possible values for the cancer argument.
    database:
        The database to query.
    cancer:
        The type of cancer.

    Returns
    -------
        The dataframe of phenotypes with patients as rows and phenotype kinds as columns.
    """
    file_path = os.path.join(data_path, database, '{}_phenotype.tsv.gz'.format(cancer))
    column_name = get_index_column_name(database, cancer, 'phenotype')
    df = pd.read_csv(file_path, compression="gzip", sep="\t", index_col=column_name)
    return df


def _load_phenotypes(data_path, database, cancer, label_name):
    # Load
    phenotype = load_phenotype(data_path, database, cancer)
    # Retrieve the column corresponding to the labels to classify.
    labels = phenotype[label_name]
    # Remove unwanted labels and associate each remaining label with a number from 0 to number of classes - 1.
    label_list = sorted(np.unique(list(labels.values)))
    label_key = clean_labels(label_list, database, cancer)
    label_map = dict(zip(label_key, range(len(label_key))))
    # inv_label_map = {v: k for k, v in label_map.items()}
    return labels, label_key, label_map


# Util

def clean_sample_IDs(data: pd.DataFrame, labels: pd.DataFrame, label_key: list):
    """
    Remove samples which are not associated with a label.

    In place modifications.

    Parameters
    ----------
    data:
        The input data where there is sample ids as index.
    labels:
        All the labels.
    label_key:
        The list of possible labels.

    Returns
    -------
        The currated list of samples.
    """

    # Extract the IDs of the samples (corresponding to individuals)
    sample_IDs = data.index.values.tolist()
    # Remove samples from methylation which are not associated with a label
    IDs_to_remove = []
    for ID in sample_IDs:
        try:
            y = labels[ID]
            if y not in label_key:  # case y is nan for example
                IDs_to_remove.append(ID)
        except KeyError:
            IDs_to_remove.append(ID)
    for ID in IDs_to_remove:
        sample_IDs.remove(ID)
    return sample_IDs


def get_index_column_name(database: Literal["PANCAN", "GDC"], cancer: Optional[Literal["BRCA"]], data_type: Literal["expression", "phenotype", "methylation"]) -> str:
    """
    Get the index column name for a given database, cancer type and data type.

    Parameters
    ----------
    database:
        Database where to look for data.
    cancer:
        Kind of cancer.
    data_type:
        The kind of data.

    Returns
    -------
        The index column name.
    """
    column = {}
    column['PANCAN-None'] = {"expression": "sample", "phenotype": "sample"}
    column['GDC-BRCA'] = {"expression": "Ensembl_ID", "phenotype": "submitter_id.samples",
                          "methylation": "Composite Element REF"}
    try:
        return column[f"{database}-{cancer}"][data_type]
    except KeyError:
        raise KeyError(
            "Please, have a look at the original files and "
            "add the name of the column corresponding to the genes IDs (expression), "
            "to the methylation sites IDs (methylation) or to the individuals IDs (phenotypes).")


def create_new_folder(path: str) -> NoReturn:
    """
    Create a folder in 'path' if it does not exist yet.

    Parameters
    ----------
    path
        Where to create the folder.
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == 17:
            pass
        else:
            raise


def read_file(file_path: str) -> list:
    """
    Return the lines of the file as a list.

    Parameters
    ----------
    file_path:
        The path of the file.

    Returns
    -------
        The lsit of stripped lines.
    """
    with open(file_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def download(url: str, file_path: str) -> NoReturn:
    """
    Download file at specified `url` to the specified `file_path`.

    If file is not compressed, it is compressed.

    Parameters
    ----------
    url:
        URL where to download the file.
    file_path:
        Path where to store the downloaded file.

    """
    # If the file already exists, we do not need to download it again.
    if os.path.isfile(file_path):
        print(file_path + ' already existing.')
    else:
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        
        # Save data.
        # If the data is not compressed, we compress it.
        if url[-3:] == '.gz':
            with open(file_path, 'wb') as f:
                f.write(data.read())
        else:
            decompressed_file_path = file_path.replace('.gz', '')
            with open(decompressed_file_path, 'wb') as f:
                f.write(data.read())
            with open(decompressed_file_path, 'rb') as f, gzip.GzipFile(file_path, 'wb') as zip_f:
                zip_f.write(f.read())
            os.remove(decompressed_file_path)
        
        # Assert that the dowload has been successful.
        if os.stat(file_path).st_size == 0:
            os.remove(file_path)
            error = IOError('Downloading {} failed.'.format(url))
            raise error


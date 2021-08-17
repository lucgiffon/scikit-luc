"""
Utility functions that are about reading or writing files/directory on the system.
"""

import errno
import hashlib
import os
import urllib.request
import numpy as np
import pathlib
import scipy.io as sio

from skluc.utils import logger
from pathlib import Path


def read_matfile(fname):
    """
    loosely copied on https://stackoverflow.com/questions/29185493/read-svhn-dataset-in-python

    Python function for importing the SVHN data set.
    """
    # Load everything in some numpy arrays
    logger.info("Read mat file {}".format(fname))
    data = sio.loadmat(fname)
    img = np.moveaxis(data['X'], -1, 0)
    lbl = data['y']
    return img, lbl

def silentremove(filename):
    """
    Remove filename without raising error if the file doesn't exist.

    :param filename: The filename
    :type filename: str
    :return: None
    """

    try:
        os.remove(filename)
        logger.debug("File {} has been removed".format(filename))
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred
        logger.debug("Directory or file {} doesn't exist".format(filename))

def create_directory(_dir, parents=True, exist_ok=True):
    """
    Try to create the directory if it does not exist.

    :param dir: the path to the directory to be created
    :return: None
    """
    logger.debug("Creating directory {} if needed".format(_dir))
    pathlib.Path(_dir).mkdir(parents=parents, exist_ok=exist_ok)

def download_file(url, directory, name=None):
    """
    Download the file at the specified url

    :param url: the end-point url of the need file
    :type url: str
    :param directory: the target directory where to download the file
    :type directory: str
    :param name: the name of the target file downloaded
    :type name: str
    :return: The destination path of the downloaded file
    """
    create_directory(directory)
    logger.debug(f"Download file at {url}")
    if name is None:
        name = os.path.basename(os.path.normpath(url))
    s_file_path = os.path.join(directory, name)
    if not os.path.exists(s_file_path):
        urllib.request.urlretrieve(url, s_file_path)
        logger.debug("File {} has been downloaded to {}.".format(url, s_file_path))
    else:
        logger.debug("File {} already exists and doesn't need to be donwloaded".format(s_file_path))

    return s_file_path

def check_file_md5(filepath, md5checksum, raise_=True):
    """
    Check if filepath checksum and md5checksum are the same

    If raise_ == True, raise an exception instead of returning False when md5sum does not correspond.

    :param filepath: The file path to check
    :param md5checksum: The checksum for verification
    :param raise_: Bool says if exception should be raised when md5 doesn't correspond.
    :return: Return True if the md5 are the same
    """
    logger.debug("Check {} md5 checksum with expected checksum {}".format(filepath, md5checksum))
    # Open,close, read file and calculate MD5 on its contents
    with open(filepath, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()
        # pipe contents of the file through
        md5_returned = hashlib.md5(data).hexdigest()
        logger.debug("Checksum of {} is {}".format(filepath, md5_returned))

    # Finally compare original MD5 with freshly calculated
    if md5checksum == md5_returned:
        logger.debug("Checksum match: file correctly downloaded")
        return True
    else:
        s_not_match = "Checksum of file {}: {} doesn't match the expected checksum {}"\
            .format(filepath, md5_returned, md5checksum)
        if raise_:
            raise ValueError(s_not_match)
        logger.debug(s_not_match)
        return False

def check_files(filepaths):
    logger.debug("Check existence of files {}".format(str(filepaths)))
    return all([os.path.exists(fpath) for fpath in filepaths])

def get_project_dir_path(__file, name_project):
    """
    Return the subpath of __file from the begining of the file to the name of the project.

    :param __file: string path to file
    :param name_project: string name of project
    :return: path to project dir
    """
    split_file = __file.split("/")
    idx_project_dir = split_file.index(name_project)
    project_dir = "/".join(split_file[:(idx_project_dir + 1)])
    project_dir = Path(project_dir)
    return project_dir
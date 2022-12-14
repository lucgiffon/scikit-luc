# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages
import io
import skluc


def read(*paths):
    """Build a file path from *paths* and return the contents."""
    with io.open(os.path.join(*paths), "r", encoding="UTF-8") as f:
        return f.read()


long_description = (
    read("README.rst")
    + "\n\n"
    + read("AUTHORS.rst")
    + "\n\n"
    + read("LICENSE.rst")
    + "\n\n"
)

print(long_description)

setup(
    # name of the package
    name="scikit-luc",
    # You can specify all the packages manually or use the find_package
    # function
    packages=find_packages(exclude=["doc", "examples"]),
    # See PEP440 for defining a proper version number
    version=str(skluc.__version__),
    # Small description of the package
    description="Science-Kit with some utilities for machine learning.",
    # Long description
    long_description=(long_description),
    # Project home page:
    url="",
    # license, author and author email
    license="GPL, Version 3",
    author="Luc Giffon",
    author_email="luc.giffon@lif.univ-mrs.fr",
    # If any packages contains data which are not python files, include them
    # package_data={'myapp': 'data/*.gif'},
    install_requires=[
        "daiquiri",
        "numpy",
        "scikit-learn",
        "numba",
        "keras",
        "scipy",
        "psutil",
        "imageio",
        "matplotlib",
        "docopt",
        "opencv-python",
        "opencv-contrib-python",
        "googledrivedownloader",
        "requests",
        "click",
        "loguru",
        "pandas",
        "pyyaml",
        "click_pathlib",
    ],
    # classifiers is needed for uploading package on pypi.
    # The list of classifiers elements can be found at :
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3 :: Only",
    ],
    # What does your project relate to?
    # keywords=['Linux', 'MacOSX', 'Windows'],
    # Platforms on which the package can be installed:
    # platforms=['Linux'],
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        "console_scripts": [
            "lazygrid=skluc.tools.lazygrid:main",
            "csvgatherer=skluc.tools.csvgatherer:main",
        ],
    },
)

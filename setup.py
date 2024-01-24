# External Includes
import os
from setuptools import find_packages, setup

# Internal Includes
from sigtorch import __version__ as VERSION


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def load_requirements():
    return read("requirements.txt").splitlines()


setup(
    name="SigTorch",
    version=VERSION,
    author="Kyle McClintick",
    author_email="kyle.mcclintick@ll.mit.edu",
    description="A set of signal analysis pipelines for pytorch",
    keywords="RF pytorch digital signal processing inference",
    url="https://llcad-github.llan.ll.mit.edu/kwmcclintick/SigTorch",
    packages=find_packages(exclude=["test*"]),
    long_description=(
        "Machine learning tools are commonly used in electrical and computer engineering systems to infer unknown parameters given a set of noisy observations. "
        " This repo is intended to provide a set of quickstart pytorch pipelines to train, predict, and evaluate low-dimensional signal data for these systems. "
        " A focus on dataset dimension generalization and parallelized training for hyperparameter optimization is considered in this repo's development."
    ),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Telecommunications Industry",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Communications :: Ham Radio",
        "Topic :: Communications :: Telephony",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=load_requirements(),
)

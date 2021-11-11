#!/usr/bin/env python

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "mftma @ git+https://github.com/schung039/neural_manifolds_replicaMFT",
    "adversarial-robustness-toolbox==1.6.1",
    "robustness @ git+https://github.com/MadryLab/robustness",
    "seaborn",
    "pandas",
    "h5py",
    "torch",
    "cvxopt",
    "pymanopt",
    "autograd",
    "requests",
    "jupyter"
]

setup(
    name='adversarial-manifolds',
    version='0.1.0',
    description="Demo code for 'Neural Population Geometry Reveals the Role of Stochasticity in Robust Perception'",
    long_description=readme,
    author="Joel Dapello, Jenelle Feather, Hang Le, Tiago Marques, SueYeon Chung",
    author_email='dapello@mit.edu',
    url='https://github.com/chung-neuroai-lab/adversarial-manifolds',
    install_requires=requirements,
    license="MIT license",
    keywords='adversarial, manifolds, neural population geometry, VOneNet, stochastic',
)

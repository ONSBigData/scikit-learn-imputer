"""Setup script for creating package from code."""
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='scikit-learn-imputer',
    version='0.1.0',
    description='Imputation tool for categorical and continuous data using scikit-learn algorithms. Includes simulation study and model persistence.',
    url='https://github.com/ONSBigData/scikit-learn-imputer',
    packages=find_packages(),
    zip_safe=False,
    install_requires=requirements
)

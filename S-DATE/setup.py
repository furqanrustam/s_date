# setup.py
from setuptools import setup, find_packages

setup(
    name='s_date',
    version='0.1',
    packages=find_packages(),
    description='A synthetic data generation library using S-DATE approach.',
    author='Furqan Rustam, Anca Delia Jurcut, and Imran Ashraf',
    author_email='furqan.rustam1@gmail.com',
    url='https://github.com/furqanrustam/s_date',     install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ],
)

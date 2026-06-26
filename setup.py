from setuptools import setup, find_packages

setup(
    name='c4f',
    version='0.1.0',
    description='Clustering for Fairness (C4F) - Evaluating machine learning model fairness without ground-truth labels.',
    author='[Author Name]',
    packages=find_packages(include=['c4f', 'c4f.*']),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scikit-learn-extra',
        'hdbscan',
        'kmodes',
        'geopy',
        'geopandas',
        'flask>=3.0'
    ],
    python_requires='>=3.8',
)

from setuptools import setup, find_packages

setup(
    name='forecasting_study',
    version='0.1.0',
    description='Example package for ML pipeline',
    packages=find_packages(include=['forecasting']),
    install_requires=[
        "pandas",
        "numpy",
        "plotnine",
        "scikit-learn"
    ],
    python_requires="==3.8.5"
)
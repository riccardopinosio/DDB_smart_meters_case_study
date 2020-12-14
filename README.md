# DDB_forecasting_case_study

Example ML pipeline for DDB master.

Please use this repository as a template for your funda product.

IMPORTANT:
the source code should reside in modules consisting of classes and functions
(no top level code) in a folder at the top level (see the forecasting folder above). This folder should contain
an empty __init__.py file so that python recognizes it as a package.
Moreover, the setup.py file should point to this folder in the packages()
line (see template). With this setup, you can create a conda environment with

```
conda create --name myenvironment python==3.8.5
```

and then do (from within your package folder):

```
conda activate myenvironment
pip install -e .
```


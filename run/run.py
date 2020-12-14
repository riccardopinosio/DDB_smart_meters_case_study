import json
from forecasting.load_data import DataLoader
# important: for the above import to work, the package needs to be
# installed in the conda environment using e.g. pip install -e .
# from the package root, or python setup.py develop.
# See https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
# for a good guide to this

def main():
    # here goes the pipeline code
    with open('./conf.json', 'r') as f:
        conf = json.load(f)
    data_loader = DataLoader(base_folder=conf['base_folder'])
    raw_data = data_loader.load_data()


if __name__ == "__main__":
    # the main function above is called when the script is
    # called from command line
    main()
'''Code to clean the data
'''
import pandas as pd
import numpy as np

class DataCleaner(object):

    def __init__(self, params):
        self.params = params

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # here goes the code to clean X
        X['col'] = X['col'].astype(str)
        # other stuff
        return X
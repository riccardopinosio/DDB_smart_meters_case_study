'''Contains classes to load data
'''
import pandas as pd
import os

class DataLoader(object):

    def __init__(self, base_folder):
        self.base_folder = base_folder

    def load_data(self):
        # load the power network data
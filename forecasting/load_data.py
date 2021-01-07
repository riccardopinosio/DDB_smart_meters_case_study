'''Contains classes to load data
'''
import pandas as pd
import numpy as np
import os

class DataLoader(object):
    def __init__(self, raw_folder):
        self.raw_folder=raw_folder
        self.power_schema = {
            "LCLid": str,
            "stdorToU": str,
            "DateTime": str, # will be converted later
            "KWH/hh (per half hour) ": str,
            "Acorn": str,
            "Acorn_grouped": str
        }

    def read_energy_df(self, x):
        df = pd.read_csv(os.path.join(self.raw_folder, x),
        dtype=self.power_schema)
        power_df = df.rename(columns={
            'KWH/hh (per half hour) ': 'kwh'
        })
        power_df['DateTime'] = pd.to_datetime(power_df['DateTime'])
        power_df['kwh'] = np.where(
            power_df['kwh'] == 'Null',
            np.nan,
            power_df['kwh']
        )
        power_df['kwh'] = power_df['kwh'].astype(float)
        return power_df

    def load_data(self):
        power_csvs = os.listdir(self.raw_folder)
        power_csvs = [x for x in power_csvs if x.startswith("Power-Networks")]

        # read into memory the dataframes with power consumption at half-hourly level
        # and process them. This way it takes a bit more time but no memory
        # errors
        power_dfs = [self.read_energy_df(x) for x in power_csvs]

        # combine the dfs into a single df
        power_df = pd.concat(power_dfs, axis=0)
        # aggregate the df of power consumption, 
        # as we only want to forecast at daily level
        housing_info = power_df[['LCLid', 'stdorToU', 'Acorn', 'Acorn_grouped']].drop_duplicates()
        power_df = power_df[['LCLid', 'DateTime', 'kwh']]
        return housing_info, power_df

# note: the following trick is useful to debug the class
# self = DataLoader(raw_folder=conf['base_folder'] + '/raw')
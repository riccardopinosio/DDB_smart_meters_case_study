'''Code to clean the data
'''
import pandas as pd
import numpy as np

class DataCleaner(object):
    def __init__(self):
        self.cleaning_statistics = {}

    def fill_gaps(self, df):
        min_date = df['date'].min()
        max_date = df['date'].max()
        date_sequence = pd.date_range(min_date, max_date)
        date_index_df = pd.DataFrame({"date": date_sequence})
        df = date_index_df.merge(df, how='left')
        # forward filling of gaps
        df = df.ffill()
        return df

    def clean_data(self, housing_info, power_df) -> list:
        '''Clean the raw data and return datasets ready to be 
        featurized
        '''
        # aggregate half-hourly power kwh to daily
        power_df['DateTime'] = power_df['DateTime'].dt.date
        power_df = power_df.rename(columns={'DateTime': 'date'})
        power_df = power_df.groupby(['date', 'LCLid']).agg(
            kwh = ('kwh', 'sum')
        ).reset_index()

        # remove everything in 2011 and last day
        power_df = (power_df
        .loc[(power_df['date'] >= pd.to_datetime('2012-01-01')) & (
            power_df['date'] <= pd.to_datetime('2014-02-27'))])

        # calculate some statistics on power_df in order to filter
        # out some households
        n_observations = (power_df.groupby('LCLid')
        .size()
        .reset_index(name='n_obs')
        .sort_values('n_obs')
        .set_index('LCLid')
        )

        gaps = (power_df.sort_values(['LCLid', 'date'])
        .groupby("LCLid")
        .apply(lambda x: x.assign(previous_date=x['date'].shift(1),
        diff = lambda x: (x['date'] - x['previous_date'])
        ))
        .reset_index(drop=True)
        )
        gaps = gaps.loc[~gaps['diff'].isnull()]
        max_gaps = (gaps.groupby('LCLid')
        .agg(max_gap=('diff', 'max'))
        )

        house_stats = pd.concat([n_observations, max_gaps],axis=1)

        # define the filters for households that need to be filtered out
        house_stats['to_keep'] = np.where(
            (house_stats['n_obs'] >= 30) &
            ( (house_stats['max_gap']  <= pd.Timedelta(days=5)) | 
            (house_stats['max_gap'].isnull())
            ),
            True,
            False)
        # now remove the households according to the house stats df
        self.house_stats = house_stats.reset_index().rename(columns={
            "index": "LCLid"
        })
        power_df = power_df.merge(
            (self.house_stats.query("to_keep == True")[['LCLid']])
            )
        power_df['date'] = pd.to_datetime(power_df['date'])
        # now pad and forward fill
        power_df = (
            power_df.sort_values(['LCLid', 'date'])
            .groupby(['LCLid'])
            .apply(self.fill_gaps)
            .reset_index(drop=True)
        )
        # clean housing info
        housing_info['Acorn_grouped'] = 'g_' + housing_info['Acorn_grouped']
        return housing_info, power_df

# self = DataCleaner()
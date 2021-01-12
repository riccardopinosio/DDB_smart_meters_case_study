import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class TransformerFitter(TransformerMixin):
    def fit(self, X=None, y=None):
        return self

class Featurizer(TransformerFitter):
    def __init__(self, housing_info):
        self.housing_info = housing_info

    def transform(self, X):
        '''Calculation of the features used to forecast the 1-day ahead
        energy consumption.
        '''
        aggregations = ['mean', 'sum', 'max', 'min']
        rolling_windows = [3, 7]
        window_features = []
        X = X.sort_values(['LCLid', 'date'])

        # calculate the rolling features for each household
        for window in rolling_windows:
            features = (
                X.groupby('LCLid')
                .rolling(window, on='date', periods=1)['kwh']
            .agg(aggregations))
            features = features.rename(columns=lambda x: x + '_' + str(window))
            window_features.append(features)
        
        # combine the rolling features
        rolling_features = pd.concat(window_features, axis=1)
        rolling_features = rolling_features.reset_index()
        # using 0 as NA flag
        rolling_features = rolling_features.fillna(-1)

        # dummify with sklearn one-hot-encoder
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        housing_info = self.housing_info
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(housing_info[['stdorToU', 'Acorn', 'Acorn_grouped']])
        self.encoder = enc
        housing_info_dummy = enc.transform(housing_info[['stdorToU', 'Acorn', 'Acorn_grouped']]).toarray()
        housing_info_dummy = pd.DataFrame(housing_info_dummy)
        cols = []
        for i in enc.categories_:
            cols = cols + i.tolist()
        housing_info_dummy.columns = cols
        housing_info_features = pd.concat(
            [housing_info[['LCLid']].reset_index(drop=True),
            housing_info_dummy.reset_index(drop=True)],
            axis=1)
        self.categorical_encoder = enc

        # this can also be done quicker with pandas.get_dummies but it has drawbacks
        # housing_info_dummy = housing_info[['stdorToU', 'Acorn', 'Acorn_grouped']].copy()
        # housing_info_dummy = pd.get_dummies(housing_info_dummy)

        # combine for final features
        final_features = rolling_features.merge(housing_info_features, on=['LCLid'], how='left')

        # now we add the target to be forecasted; this is the consumption at date D + 1
        X['kwh_1'] = (X.groupby('LCLid')['kwh'].shift(-1))
        final_features = final_features.merge(X[['LCLid', 'date', 'kwh_1']],
        how='left')
        return final_features

# self = Featurizer(housing_info = housing_info)


        
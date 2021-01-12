import pandas as pd
import numpy as np
import itertools
from copy import deepcopy
from sklearn.metrics import mean_squared_error

class Hypertuner(object):
    def __init__(self, estimator, tuning_params, validation_mapping):
        self.estimator = estimator
        self.tuning_params = tuning_params
        self.validation_mapping = validation_mapping

    def calculate_mean_cv_error(self, train_set, estimator_cv):
        # now perform cross validation fitting for each split
        splits = train_set['cv_split'].unique().tolist()
        splits.sort()

        cv_errors = []

        for i in splits:
            train_split = train_set.query(f"cv_split != {i}")
            X_train = train_split.drop(['LCLid', 'date', 'kwh_1', 'cv_split'],axis=1)
            y_train = train_split['kwh_1']
            estimator_cv.fit(X=X_train, y = y_train)
            # evaluate the model on split 1
            test_obs = train_set.query(f"cv_split == {i}")
            X_test = test_obs.drop(['LCLid', 'date', 'kwh_1', 'cv_split'],axis=1)
            y_pred = estimator_cv.predict(X_test)
            # calculate error measure on this fold for the estimator with the
            # given parameters
            rmse = np.sqrt(np.sum(np.square(X_test['kwh_1'] - y_pred))/X_test.shape[0])
            np.sqrt(mean_squared_error(X_test['kwh_1'], y_pred))
            cv_errors.append(rmse)

        mean_rmse = mean(cv_errors)
        return mean_rmse

    def tune_model(self, train_set):
        '''Perform the hypertuning of the estimator on the train set
        for all the combinations of the hyperparameters
        '''
        parameter_combos = []
        parameter_combos_dicts = []

        for a in itertools.product(*self.tuning_params.values()):
            parameter_combos.append(a)

        for i in parameter_combos:
            d = {}
            for j in range(len(i)):
                d[list(self.tuning_params.keys())[j]] = i[j]
            parameter_combos_dicts.append(d)

        validation_mapping_train = self.validation_mapping.query("test == False")
        train_set = train_set.merge(validation_mapping_train[['LCLid', 'date', 'cv_split']])
        cv_errors = []

        for d in parameter_combos_dicts:
            estimator_cv = deepcopy(self.estimator)
            estimator_cv = estimator_cv.set_params(**d)
            mean_cv_error = self.calculate_mean_cv_error(train_set, estimator_cv)
            cv_errors.append(mean_cv_error)


estimator = RandomForestRegressor(random_state=1234)
tuning_params = conf["training_params"]["hypertuning"]["RF_params"]

self = Hypertuner(estimator = RandomForestRegressor(random_state=1234),
tuning_params = conf["training_params"]["hypertuning"]["RF_params"],
validation_mapping=validation_mapping)
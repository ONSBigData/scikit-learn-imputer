"""
SklearnImputer

Created by G Bettsworth 09/2020

This tool can be used to impute missing values in a data frame with mixed variable types using a scikit-learn classifier and regressor.

The benefits of the tool are:
- the wrangling of the data is internal to the module, so the input data can include columns in string format. Features are
created internally including one-hot encoding.
- Any scikit-learn classifier and regressor can be used. This makes many imputation methods available, from common tools
such as k-nearest neighbour and regression, as well as more advanced tools such as multi-layer perceptron neural network and random
forests.
- The tool also has the ability to save trained models and perform imputation using a saved model. This is helpful for methods
which can take a long time to train but are fast to implement once trained (e.g. neural network).
- There is also the option to train the model each time imputation is performed - which is helpful for models that take up a lot
of memory if saved, have limited time benefits from training and are deterministic (e.g. k-nearest neighbour).
- The tool includes a function called 'select_model' which allows you to assess different methods and hyper-parameters in terms
of accuracy and timeliness.
"""
import copy
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.impute import MissingIndicator
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class SklearnImputer():
    """
    A tool to impute missing values in data frames with mixed types (both categorical and continuous).
    Any sklearn classifier and regressor can be used to impute. Index is used as the ID and must be numeric.

    Parameters
    ----------
    input_data : pandas.DataFrame
            DataFrame containing the missing values and any additional features.

    categorical : list, default=None
            List of categorical variables in the data frame. If not in list, the variable is assumed to be continuous.

    save_models_to : string (default=r'./saved_model.z')
            This is a file directory location where the model will be saved (if fit is run and transform=False).
            This file must be a .z file and the model will be saved as a bytes object. To view the model
            read it in using joblib.load(save_models_to) in python.

    round_column : list, optional (default=None)
            List of continuous variables that are to be rounded as part of the model

    class_threshold : int, optional (default = 30)
                If a categorical variable has more classes than this threshold then it will not be used as a feature in the model.

    features: list, (default = None)
        list of columns to use as features, default all are used

    missing_flags: boolean, (default = False)
        Include the missing flags as features
    """

    def __init__(self, input_data, categorical=None, save_models_to=r'./saved_model.z', round_column=None,
                 class_threshold=30, features=None, include_missing_flags=False):

        # default arguments should be immutable hence 'None' input
        round_column = [] if round_column is None else round_column
        features = [] if features is None else features
        categorical = [] if categorical is None else categorical

        function_inputs = [input_data, categorical, save_models_to, round_column,
                           class_threshold, features, include_missing_flags]

        string_versions = ['input_data', 'categorical', 'save_models_to', 'round_column',
                           'class_threshold', 'features', 'include_missing_flags']

        expected_types = [pd.DataFrame, list, str, list, int, list, bool]

        for parameter, input, expected_type in zip(string_versions, function_inputs, expected_types):
            if not isinstance(input, expected_type):

                input_type = str(type(input))

                for string in ['class', '<', '>', ' ', "'"]:
                    input_type = input_type.replace(string, '')
                    expected_type = str(expected_type).replace(string, '')

                raise TypeError(f"{parameter} must be {expected_type} not {input_type}")

        not_in_df = [x for x in categorical if x not in input_data.columns]
        if len(not_in_df) != 0:
            raise KeyError(f'The following columns are not in data frame: {not_in_df}')

        if not os.path.exists(os.path.dirname(save_models_to)):
            raise OSError('The directory specified in save_models_to does not exist')

        self.input_data = input_data
        self.categorical = categorical
        self.save_models_to = save_models_to
        self.round_column = round_column
        self.class_threshold = class_threshold
        self.features = features
        self.include_missing_flags = include_missing_flags

    def missing_indicator(self):
        """
        Returns the output of sklearn.impute.MissingIndicator as a pandas DataFrame
        """

        return pd.DataFrame(MissingIndicator(features='all').fit_transform(self.input_data),
                            index=self.input_data.index, columns=[x + '_flag' for x in self.input_data.columns])

    def missing_metrics(self):
        """
        Returns information about the missing data.

        Returns
        -------
        pandas.DataFrame
            variable is the column into the input_data frame, observed is the number of
            non-missing values in that column, missing is the number of missing values in that
            column, total is the total number of rows, and the imputation rate is the
            number of missing values divided by the number of rows.
        """
        missing_flags = self.missing_indicator()

        columns = [s.replace('_flag', '') for s in missing_flags.columns]
        total = list(missing_flags.count())
        missing = list(missing_flags.sum())
        observed = [int(t) - int(i) for i, t in zip(missing, total)]
        rate = [int(i) / int(t) for i, t in zip(missing, total)]

        return pd.DataFrame({'variable': columns, 'observed': observed, 'missing': missing,
                             'total': total, 'imputation_rate': rate})

    def create_features(self):
        """
        Function to create the feature spaces to be used for the models.

        Simple hierarchical impute, then categories are one-hot encoded and the missing flags are merged.

        Returns
        -------
        Output data frame
        """

        # simple hierarchical impute so that we don't need to worry about missing values

        df = self.input_data[self.features] if len(self.features) > 0 else self.input_data
        categorical = [x for x in self.categorical if x in self.features] if len(
            self.features) > 0 else self.categorical

        simple_impute = df.fillna(method='ffill')
        simple_impute = simple_impute.fillna(method='bfill')

        too_many_classes = [x for x in categorical if df[x].nunique() > self.class_threshold]
        features_to_keep = [x for x in categorical if x not in too_many_classes]

        simple_impute.drop(labels=too_many_classes, axis=1, inplace=True)

        one_hot = pd.get_dummies(data=simple_impute, columns=features_to_keep)

        if self.include_missing_flags:
            missing_flags = self.missing_indicator()
            missing_flags = missing_flags.astype(int)

            all_features = pd.merge(one_hot, missing_flags, right_index=True, left_index=True)

        else:
            all_features = one_hot

        return all_features

    def fit(self, classification, regression, transform=False, test_size=0.1, random_seed=42, scaler=MinMaxScaler()):
        """
        Scaling is done within fit to avoid using model persistence more than once in the class.
        Training a model per variable. This returns the trained model and the test performance.

        Parameters
        -----------
        classification : sklearn classifier

        regression : sklearn regressor

        test_size : int, optional (default = 0.2)
                The size of the test data

        scaler : class, optional (default = MinMaxScaler)
                sklearn scaler

        random_seed : int, optional (default = 42)
        """
        global imputed
        start_overall_time = time.time()

        all_features = self.create_features()

        fitted_scaler = scaler.fit(all_features)

        features = pd.DataFrame(fitted_scaler.fit_transform(all_features), index=all_features.index,
                                columns=all_features.columns)

        missing_info = self.missing_metrics()
        missing_info = missing_info[missing_info['missing'] > 0]
        to_impute = list(missing_info['variable'])

        trained_models = {}

        if transform:
            imputed = self.input_data.copy()

        for column in to_impute:
            print('-------------------------------')
            print(f'Training model for {column}')

            start_time = time.time()

            target = self.input_data[column]
            target = target.fillna(-1234)
            not_missing = target[target != -1234]

            if column in self.categorical:
                fitted_label_encode = LabelEncoder().fit(not_missing)
                not_missing = pd.Series(fitted_label_encode.transform(not_missing), index=not_missing.index)
            else:
                fitted_label_encode = None

            train_targets, test_targets = train_test_split(not_missing, test_size=test_size, random_state=random_seed)

            # emulate test train split in features
            features_data = features.copy()
            drop_columns = [x for x in features_data.columns if column in x]
            features_data.drop(labels=drop_columns, axis=1, inplace=True)
            train_features = features_data.loc[list(train_targets.index)]
            test_features = features_data.loc[list(test_targets.index)]

            model = classification if column in self.categorical else regression

            print(f'Model: {model}')

            fitted_model = model.fit(train_features, train_targets)
            train_time = time.time() - start_time

            start_test_timer = time.time()
            print('Testing model')
            predictions = fitted_model.predict(test_features)
            predictions = np.round(predictions) if column in self.round_column else predictions
            test_time = time.time() - start_test_timer

            if model == regression:
                model_performance = {'mse': mean_squared_error(predictions, test_targets)}
            else:
                model_performance = {'accuracy': accuracy_score(predictions, test_targets)}

            print(model_performance)

            if transform:
                start_impute_timer = time.time()
                print('Imputing missing values')
                na = target[target == -1234]
                impute_features = features_data.loc[list(na.index)]
                predictions = fitted_model.predict(impute_features)
                predictions = np.round(predictions) if column in self.round_column else predictions

                predictions = fitted_label_encode.inverse_transform(
                    predictions) if column in self.categorical else predictions

                imputed.loc[list(na.index), column] = list(predictions)

                impute_time = time.time() - start_impute_timer

                trained_models[column] = {'impute_time': impute_time,
                                          'trained_model': fitted_model,
                                          'model_features': list(features_data.columns),
                                          'label_encoder': fitted_label_encode, 'train_time': train_time,
                                          'test_time': test_time,
                                          'model_performance': model_performance}

            else:

                trained_models[column] = {'trained_model': copy.deepcopy(fitted_model),
                                          'model_features': list(features_data.columns),
                                          'label_encoder': copy.deepcopy(fitted_label_encode),
                                          'train_time': train_time, 'test_time': test_time,
                                          'model_performance': model_performance}

            print('-------------------------------')

        trained_models['global_scaler'] = fitted_scaler

        joblib.dump(trained_models, self.save_models_to) if not transform else None

        if transform:
            trained_models['imputed_data'] = imputed

        trained_models['overall_time'] = time.time() - start_overall_time

        return trained_models

    def transform(self):
        """
        Performs imputation using the saved model in the specified location (self.save_models_to).
        :return:
        """

        start_overall_time = time.time()

        trained_models = joblib.load(self.save_models_to)

        to_impute = list(trained_models.keys())
        to_impute = [x for x in to_impute if x in self.input_data.columns]

        all_features = self.create_features()

        fitted_scaler = trained_models['global_scaler']

        features = pd.DataFrame(fitted_scaler.fit_transform(all_features), index=all_features.index,
                                columns=all_features.columns)

        impute_times = {}
        imputed = self.input_data.copy()

        for column in to_impute:
            start_impute_timer = time.time()
            print('-------------------------------')
            print(f'Imputing missing_values for {column}')

            target = self.input_data[column]
            target = target.fillna(-1234)
            na = target[target == -1234]
            features_data = features.copy()
            impute_features = features_data.loc[list(na.index)]
            keep_features = trained_models[column]['model_features']
            impute_features = impute_features[keep_features]

            fitted_model = trained_models[column]['trained_model']
            fitted_label_encode = trained_models[column]['label_encoder']

            predictions = fitted_model.predict(impute_features)
            predictions = np.round(predictions) if column in self.round_column else predictions
            predictions = fitted_label_encode.inverse_transform(
                predictions) if column in self.categorical else predictions

            imputed.loc[list(na.index), column] = list(predictions)

            impute_time = time.time() - start_impute_timer

            impute_times[column] = impute_time

            print('-------------------------------')

        overall_time = time.time() - start_overall_time

        return {'imputed_data': imputed, 'impute_times': impute_times, 'overall_time': overall_time}

    def validate(self, validation_set):
        """
        Validate an existing model using a validation set.
        """

        start_overall_time = time.time()

        trained_models = joblib.load(self.save_models_to)

        to_impute = list(trained_models.keys())
        to_impute = [x for x in to_impute if x in validation_set.columns]

        all_features = self.create_features()

        fitted_scaler = trained_models['global_scaler']

        features = pd.DataFrame(fitted_scaler.fit_transform(all_features), index=all_features.index,
                                columns=all_features.columns)

        output = {}
        imputed = validation_set.copy()

        for column in to_impute:
            start_impute_timer = time.time()
            print('-------------------------------')
            print(f'Testing values for {column}')

            target = self.input_data[column]
            target = target.fillna(-1234)
            not_missing = target[target != -1234]
            features_data = features.copy()
            impute_features = features_data.loc[list(not_missing.index)]
            keep_features = trained_models[column]['model_features']
            impute_features = impute_features[keep_features]

            fitted_model = trained_models[column]['trained_model']
            fitted_label_encode = trained_models[column]['label_encoder']

            if column in self.categorical:
                try:
                    not_missing = pd.Series(fitted_label_encode.transform(not_missing), index=not_missing.index)
                except ValueError as v:
                    print(f'WARNING: {v}')
                    print('Accuracy will be 0')

            predictions = fitted_model.predict(impute_features)
            predictions = np.round(predictions) if column in self.round_column else predictions

            if column not in self.categorical:
                model_performance = {'mse': mean_squared_error(predictions, not_missing)}
                print(model_performance)
            else:
                model_performance = {'accuracy': accuracy_score(predictions, not_missing)}
                print(model_performance)

            test_time = time.time() - start_impute_timer

            output[column] = {'model_performance': model_performance, 'test_time': test_time}

            print('-------------------------------')

        overall_time = time.time() - start_overall_time

        return {'overall_time': overall_time, 'tests': output}

    def select_model(self, options, path=None):
        """
        Select model can be used to input various models and see how they compare in terms of accuracy and timeliness.
        Includes the imputation stage as the speed of the imputation is an important output metric.

        Parameters
        ----------

        options : list of dictionaries
                This should be a list of potential models to consider. The layout should be [{'classification': sklearn.model,
                'regression': sklearn.model, 'check_separate_fit_transform_time': False}, ect...]
                check_separate_fit_transform_time returns the time differences between including the imputation (or 'transform') stage
                in the fit method or doing the fit and transform separately. This helps decide whether it is worth retraining
                the model every time you do imputation or saving the model, and reading it (using joblib) in for each time imputation is done.
                For knn, check_separate_fit_transform_time should always be False as joblib for knn causes a MemoryError.
                
        path : string, optional (default = None)
                Path to save results after each iteration, in case it fails on a big run.

        Returns
        --------
        pd.DataFrame
            DataFrame summarising the results of the simulation study with the following columns
            'model': scikit-learn models with parameters used,
            'fit_function_time': total time taken to complete the fit method (training) in
            SklearnImputer (nan if 'check_separate_fit_transform_time': False,
            'user_rollout_time_transform': total time taken to complete the transform method (imputation) in
            SklearnImputer (nan if 'check_separate_fit_transform_time': False,
            'user_rollout_time_fit_transform': total time taken to complete the fit(transform=True) method
            (training + imputation) in SklearnImputer
             mse_scores/accuracy_scores for each imputed column: the mean squared error for continuous columns
             and accuracy score for categoricals between the imputed values and true values
            'model_size': the size of the model in bytes when saved via joblib
            (nan if 'check_separate_fit_transform_time': False)
            ranks: ranking of performances where lower numbers are better
            'sum_of_performance_ranks': summation of the mse_scores/accuracy score ranks, lower numbers are better.
        """

        missing_info = self.missing_metrics()
        missing_info = missing_info[missing_info['missing'] > 0]
        to_impute = list(missing_info['variable'])

        mse_scores = [x + '_mse' for x in to_impute if x not in self.categorical]
        accuracy_scores = [x + '_accuracy' for x in to_impute if x in self.categorical]

        model_performance_summary = pd.DataFrame(columns=['model', 'fit_function_time', 'user_rollout_time_transform',
                                                          'user_rollout_time_fit_transform'] + mse_scores + accuracy_scores + [
                                                             'model_size'])

        for x in range(0, len(options)):
            print('***********************************')
            print(f'Model: {options[x]}')

            model = str(options[x])
            # we want to avoid pickle for knn as this can cause memory problems
            if options[x]['check_separate_fit_transform_time']:
                trained_models = self.fit(classification=options[x]['classification'],
                                          regression=options[x]['regression'])
                imputed = self.transform()

                model_size = os.path.getsize(self.save_models_to)

            # due to the importance of roll-out time, we want to see how fast fit transform is compared to two separate stages
            ft_trained_models = self.fit(classification=options[x]['classification'],
                                         regression=options[x]['regression'], transform=True)

            mse_scores_input = []
            accuracy_scores_input = []

            for column in to_impute:
                # used .get instead of indexing due to KeyErrors
                column_dictionary = ft_trained_models.get(column)
                model_performance = column_dictionary.get('model_performance')

                if column not in self.categorical:
                    mse_scores_input.append(model_performance.get('mse'))
                else:
                    accuracy_scores_input.append(model_performance.get('accuracy'))

            if options[x]['check_separate_fit_transform_time']:
                model_performance_summary.loc[len(model_performance_summary)] = [model,
                                                                                 trained_models.get('overall_time'),
                                                                                 imputed.get('overall_time'),
                                                                                 ft_trained_models.get(
                                                                                     'overall_time')] + mse_scores_input + \
                                                                                accuracy_scores_input + [model_size]
            else:
                model_performance_summary.loc[len(model_performance_summary)] = [model, np.nan,
                                                                                 np.nan,
                                                                                 ft_trained_models.get(
                                                                                     'overall_time')] + mse_scores_input + \
                                                                                accuracy_scores_input + [np.nan]

            # as this function takes a long time to run, this can be saved in case of an error late on
            if path is not None:
                model_performance_summary.to_csv(path, index=False)

        # TODO - find out what the error would be
        try:
            model_performance_summary['time_saved_by_saved_model'] = model_performance_summary[
                                                                         'user_rollout_time_fit_transform'] - \
                                                                     model_performance_summary[
                                                                         'user_rollout_time_transform']
        except Exception:
            model_performance_summary['time_saved_by_saved_model'] = np.nan

        for column in ['fit_function_time', 'user_rollout_time_transform',
                       'user_rollout_time_fit_transform'] + mse_scores + ['model_size']:
            # lower scores are better
            model_performance_summary[f'{column}_rank'] = model_performance_summary[column].rank(
                ascending=True)

        for column in accuracy_scores:
            # higher scores are better
            model_performance_summary[f'{column}_rank'] = model_performance_summary[column].rank(ascending=False)

        performance_ranks = [x + '_rank' for x in mse_scores] + [x + '_rank' for x in accuracy_scores]
        model_performance_summary['sum_of_performance_ranks'] = model_performance_summary[
            performance_ranks].sum(
            axis=1)

        if path is not None:
            model_performance_summary.to_csv(path, index=False)

        print('***********************************')

        return model_performance_summary


def find_imputation_variance(dictionary_of_data_frames, categorical):
    """
    Function to find imputation variance
    """
    combined = pd.concat(dictionary_of_data_frames, axis=1)

    combined.columns = combined.columns.droplevel(0)

    continuous = [x for x in combined.columns if x not in categorical]
    continuous_df = pd.DataFrame(combined[continuous].mean())
    continuous_df['Variable'] = continuous_df.index

    all_vs = continuous_df

    for column in categorical:

        for x in list(dictionary_of_data_frames.keys()):
            value, counts = np.unique(dictionary_of_data_frames[x][column], return_counts=True)
            objects_entropy = entropy(counts)

            all_vs.loc[len(all_vs)] = [objects_entropy, column]

    imp_variance = pd.DataFrame(all_vs.groupby('Variable').var())
    imp_variance = imp_variance.reset_index(drop=False)

    imp_variance.columns = ['variable', 'imputation_variance']

    return imp_variance

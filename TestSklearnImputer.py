"""
Tests for SklearnImputer

By G Bettsworth 09/2020

Note use unittest methods for assertion in order to get useful error messages.

To run from command line: python -m unittest -v  modules.TestSklearnImputer
"""
import copy
import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.testing import ignore_warnings
from tabulate import tabulate

from HierarchicalHotDeck import HierarchicalHotDeck
from SklearnImputer import SklearnImputer

input_data = pd.read_csv(r'./modules/test_data/territories.csv')
categorical_list = ['Name', 'Location']
random_state = 327

# block prints for tests to simplify output
sys.stdout = open(os.devnull, 'w')

warnings.simplefilter(action='ignore', category=FutureWarning)


class TestClassMethod(unittest.TestCase):
    """
    Tests for the __init__ method.
    """

    def test_no_error(self):
        """
        Test that when parameters are correct and defaults specified it runs without error.
        """
        sk_imp = SklearnImputer(input_data=input_data,
                                categorical=categorical_list)

        self.assertTrue('sk_imp' in locals())


class TestErrorLogging(unittest.TestCase):
    """
    Tests for the __init__ method.
    """

    def test_type_errors(self):
        """
        Test that the type errors are successfully raised for every parameter
        """
        string_versions = ['input_data', 'categorical', 'save_models_to', 'round_column',
                           'class_threshold', 'features', 'include_missing_flags']

        expected_types = ['pandas.core.frame.DataFrame', 'list', 'str', 'list', 'int', 'list', 'bool']

        base_parameters = [input_data, categorical_list, r'./saved_model.z', [], 30, [], False]

        parameter_sets = []

        for list_index in range(len(base_parameters)):
            parameter_set = copy.deepcopy(base_parameters)
            parameter_set[list_index] = {'a': 1, 'b': 2}
            parameter_sets.append(parameter_set)

        for parameter, expected, input_parameters in zip(string_versions, expected_types, parameter_sets):
            error = "None"

            try:
                SklearnImputer(*input_parameters)
            except TypeError as e:
                error = str(e)

            self.assertEqual(error, f"{parameter} must be {expected} not dict")

    def test_key_error(self):
        """
        Test that KeyError is raised when incorrect columns are specified in 'categorical'
        """
        error = "None"

        try:
            SklearnImputer(input_data=input_data, categorical=['pilot', 'space'])
        except KeyError as e:
            error = str(e)

        self.assertEqual(error, '"The following columns are not in data frame: [\'pilot\', \'space\']"')

    def test_os_error(self):
        """
        Test that KeyError is raised when incorrect columns are specified in 'categorical'
        """
        error = "None"

        try:
            SklearnImputer(input_data=input_data, save_models_to=r'./folder_that_does_not_exist/test.z')
        except OSError as e:
            error = str(e)

        self.assertEqual(error, 'The directory specified in save_models_to does not exist')


class TestMissingIndicator(unittest.TestCase):
    """
    Tests the missing indicator
    """

    def test_expected_output(self):
        """
        Test the result is expected with a standard input
        """
        input_data = pd.DataFrame({'a': ['A', np.nan, 'B'],
                                   'b': [1, np.nan, 2]})

        expected_data = pd.DataFrame({'a_flag': [False, True, False],
                                      'b_flag': [False, True, False]})

        output = SklearnImputer(input_data=input_data, categorical=['a']).missing_indicator()

        assert_frame_equal(output, expected_data)

    def test_merge(self):
        """
        Test the missing output can be merged with input_data
        """
        input_data = pd.DataFrame({'a': ['A', np.nan, 'B'],
                                   'b': ['C', np.nan, 'D']}, index=[101, 102, 103])

        expected_data = pd.DataFrame({'a': ['A', np.nan, 'B'],
                                      'b': ['C', np.nan, 'D'],
                                      'a_flag': [False, True, False],
                                      'b_flag': [False, True, False]}, index=[101, 102, 103])

        output = SklearnImputer(input_data=input_data, categorical=['a']).missing_indicator()

        merged_data = pd.merge(input_data, output, right_index=True, left_index=True)

        assert_frame_equal(merged_data, expected_data)


class TestMetrics(unittest.TestCase):
    """
    Test that the missing metrics ae as expected
    """

    def test_expected_output(self):
        input_data = pd.DataFrame({'a': ['A', np.nan, 'B', 'C'],
                                   'b': [1, np.nan, np.nan, 4]})

        output = SklearnImputer(input_data=input_data, categorical=['a']).missing_metrics()

        expected = pd.DataFrame({'variable': ['a', 'b'], 'observed': [3, 2], 'missing': [1, 2],
                                 'total': [4, 4], 'imputation_rate': [0.25, 0.5]})

        assert_frame_equal(output, expected)


class TestFeatures(unittest.TestCase):
    """
    Test create_features method
    """

    def __init__(self, *args, **kwargs):
        """
        Creates features for territories data frame for testing purposes
        """

        super(TestFeatures, self).__init__(*args, **kwargs)

        self.output = SklearnImputer(input_data=input_data, categorical=categorical_list,
                                     class_threshold=14).create_features()

    def test_columns_as_expected(self):
        """
        Checks column headings are correct.
        Note that 'Name' has been excluded as a feature using class_threshold
        """

        self.assertListEqual(list(self.output.columns),
                             ['km2', 'gdp_per_capita', 'population', 'Location_Antartica',
                              'Location_Caribbean', 'Location_Europe', 'Location_Indian_Ocean',
                              'Location_Mid_Atlantic', 'Location_Oceania',
                              'Location_South_Atlantic'])

    def test_non_missing_floats_unaffected(self):
        """
        Test that columns that have no missing values and are floats are exactly the same after
        transformation
        """
        assert_series_equal(input_data['population'], self.output['population'])


class TestFitAndFitTransform(unittest.TestCase):
    """
    Test the fit method with transform=False and transform=True, and the transform method.
    """

    def __init__(self, *args, **kwargs):
        """
        Run sklearn methods for input_data to obtain outputs.
        Note: convergence warning switched off for MLP.
        HierarchicalHotDeck raises error
        """
        super(TestFitAndFitTransform, self).__init__(*args, **kwargs)

        classifiers = [HierarchicalHotDeck(),
                       DummyClassifier(strategy='most_frequent'),
                       DummyClassifier(strategy='stratified', random_state=random_state),
                       LogisticRegression(max_iter=1, random_state=random_state),
                       DecisionTreeClassifier(random_state=random_state),
                       RandomForestClassifier(random_state=random_state),
                       KNeighborsClassifier(),
                       ExtraTreesClassifier(random_state=random_state),
                       MLPClassifier(random_state=random_state)]

        regressors = [HierarchicalHotDeck(),
                      DummyRegressor(strategy='mean'),
                      DummyRegressor(strategy='median'),
                      LinearRegression(),
                      DecisionTreeRegressor(random_state=random_state),
                      RandomForestRegressor(random_state=random_state),
                      KNeighborsRegressor(),
                      ExtraTreesRegressor(random_state=random_state),
                      MLPRegressor(random_state=random_state)]

        name = ['hh', 'dummy1', 'dummy2', 'regression', 'decision_tree', 'random_forest', 'knn', 'extra_trees', 'mlp']

        sk_imp = SklearnImputer(input_data=input_data, categorical=categorical_list, round_column=['population'],
                                class_threshold=14)

        self.tested = {}

        for classification, regression, name in zip(classifiers, regressors, name):
            with ignore_warnings(category=ConvergenceWarning):
                fit = sk_imp.fit(classification=classification, regression=regression)
                transform = sk_imp.transform()
                fit_transform = sk_imp.fit(classification=classification, regression=regression, transform=True)

                self.tested[name] = {'fit': fit, 'transform': transform, 'fit_transform': fit_transform}

        os.remove(r'./saved_model.z')

    def test_imputation_of_missing_values_transform(self):
        """
        Test that the imputation returns data frame with missing values imputed for transform method.
        If it fails, output showing the full test data frame is printed.
        """

        for name in list(self.tested.keys()):
            transform = self.tested[name]['transform']['imputed_data']

            try:
                self.assertFalse(transform.isnull().values.any())
            except AssertionError:
                sys.stdout = sys.stdout = sys.__stdout__
                raise AssertionError("DataFrame contains missing values: \n"
                                     f"{tabulate(transform)}")

    def test_imputation_of_missing_values_fit_transform(self):
        """
        Test that the imputation returns data frame with missing values imputed for fit(transform=True) method.
        If it fails, output showing the full test data frame is printed.
        """

        for name in list(self.tested.keys()):
            fit_transform = self.tested[name]['fit_transform']['imputed_data']

            try:
                self.assertFalse(fit_transform.isnull().values.any())
            except AssertionError:
                sys.stdout = sys.stdout = sys.__stdout__
                raise AssertionError("DataFrame contains missing values: \n"
                                     f"{tabulate(fit_transform)}")


if __name__ == '__main__':
    unittest.main()
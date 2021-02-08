"""
Hierarchical Hotdeck prediction

By G Bettsworth
"""
import itertools
import string
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


class HierarchicalHotDeck():
    def __init__(self, select_from_model=None, use='all'):
        self.select_from_model = select_from_model
        self.use = use

    def fit(self, x, Y):
        if self.use == 'features':
            self.var_selection = SelectFromModel(estimator=self.select_from_model).fit(x, Y)
            features = self.var_selection.transform(x)

        if self.use == 'dimension_reduction':
            self.var_selection = self.select_from_model.fit(x)
            features = self.var_selection.transform(x)

        if self.use == 'all':
            features = x

        targets = pd.DataFrame(Y)
        features = pd.DataFrame(features)

        self.data = pd.merge(targets, features, right_index=True, left_index=True)

        alphabet_list = list(string.ascii_lowercase) + list(string.ascii_uppercase)

        self.cols = itertools.islice(alphabet_list, len(list(self.data.columns)))

        self.data.columns = self.cols

        return self

    def predict(self, X):
        if self.use == 'features' or self.use == 'dimension_reduction':
            features = self.var_selection.transform(X)
        else:
            features = X

        features_cols = [x for x in self.data.columns if x != 'a']

        features = pd.DataFrame(features, columns=features_cols)

        features['a'] = np.nan

        df = features.append(self.data, ignore_index=True, sort=True)
        df = df.reset_index(drop=True)

        index = df['a'].index[df['a'].apply(np.isnan)]

        df = df.sort_values(features_cols)
        df['a'] = df['a'].ffill()
        df['a'] = df['a'].bfill()

        predictions = df.loc[list(index), 'a']

        output = predictions.to_numpy()
        # if no numbers after the decimal place then return integer output
        output = output.astype(int) if np.isclose(output, np.round(output)).all() else output

        return output


class TestHierarchicalHotDeck(unittest.TestCase):

    def test_scikit_learn_classifier_compatibility(self):
        """
        Simple test for the class to check that it is compatible with scikit-learn classification.
        """

        X, y = make_classification(random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        model = HierarchicalHotDeck(select_from_model=LogisticRegression(max_iter=1, random_state=0))
        fitted_model = model.fit(X_train, y_train)
        output = fitted_model.predict(X_test)

        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1])

        assert_array_equal(x=output, y=expected_output)
        self.assertEqual(accuracy_score(output, y_test), 0.56)

    def test_scikit_learn_regressor_compatibility(self):
        """
        Simple test for the class to check that it is compatible with scikit-learn regression.
        """

        X, y = make_regression(random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        model = HierarchicalHotDeck(select_from_model=PCA(n_components=1, random_state=0),
                                    use='dimension_reduction')

        fitted_model = model.fit(X_train, y_train)
        output = fitted_model.predict(X_test)
        expected_output = np.array([-165.646056, 44.43178627, 57.0661767, -155.40695056, -79.24125662,
                                    -102.36458935, -155.67806246, 58.99814991, -151.47230091, 58.99814991,
                                    58.99814991, -166.45918392, 55.93824886, -3.38549645, -155.67806246,
                                    -151.47230091, -31.2184318, -151.47230091, -84.02671899, -148.60915088,
                                    -78.74149031, 319.32505752, -219.07412944, -165.646056, -84.02671899])

        assert_array_equal(x=np.round(output, 2), y=np.round(expected_output, 2))
        self.assertEqual(np.round(mean_squared_error(output, y_test), 2), 38257.87)


if __name__ == '__main__':
    unittest.main()

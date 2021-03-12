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
from sklearn.metrics import accuracy_score


class HierarchicalHotDeck:
    """
    Prediction by sorting the data set.
    """
    def fit(self, x, Y):

        variables = [x, Y]
        variable_names = ['x', 'Y']
        types = [np.ndarray]*2

        for variable, name, variable_type in zip(variables, variable_names, types):
            if not isinstance(variable, variable_type):
                raise TypeError(f"The variable {name} needs to be a {variable_type}")

        targets = pd.DataFrame(Y.transpose())
        features = pd.DataFrame(x)

        self.data = pd.merge(targets, features, right_index=True, left_index=True)

        alphabet_list = list(string.ascii_lowercase) + list(string.ascii_uppercase)

        self.cols = itertools.islice(alphabet_list, len(list(self.data.columns)))

        self.data.columns = self.cols

        return self

    def predict(self, X):

        if not isinstance(X, np.ndarray):
            raise TypeError(f"The variable X needs to be an ndarray")

        features_cols = [x for x in self.data.columns if x != 'a']

        features = pd.DataFrame(X, columns=features_cols)

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
    """
    Regression and classification work the same.
    """

    def test_hierarchical_hotdeck(self):

        train = np.array([[2, 3], [1, 6], [4, 5], [3, 4], [4, 6]])
        train_targets = np.array([1, 1, 1, 0, 0])
        test = np.array([[6, 2], [0, 1], [5, 7]])
        test_targets = np.array([0, 1, 0])

        model = HierarchicalHotDeck()
        fitted_model = model.fit(train, train_targets)
        output = fitted_model.predict(test)

        expected_output = test_targets

        assert_array_equal(x=output, y=expected_output)
        self.assertEqual(accuracy_score(output, test_targets), 1)


if __name__ == '__main__':
    unittest.main()

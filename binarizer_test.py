import unittest
import numpy as np
import pandas as pd

from features_binarizer import FeaturesBinarizer
from sklearn.preprocessing import OneHotEncoder


class Test(unittest.TestCase):

    def test_FeaturesBinarizer(self):
        """Test FeaturesBinarizer
        """
        # Create a features matrix
        values = np.array([[0.00902084, 0.54159776, 0., 3.],
                           [0.46599565, -0.71875887, 0., 2.],
                           [0.52091721, -0.83803094, 1., 2.],
                           [0.47315496, 0.0730993, 1., 1.],
                           [0.08180209, -1.11447889, 0., 0.],
                           [0.45011727, -0.57931684, 0., 0.],
                           [2.04347947, -0.10127498, 1., 7.],
                           [-0.98909384, 1.36281079, 0., 0.],
                           [-0.30637613, -0.19147753, 1., 1.],
                           [0.27110903, 0.44583304, 0., 0.]])
        columns = ['c:continuous', 'a:continuous', 'd', 'b:discrete']
        features = pd.DataFrame(values, columns=columns)

        # 1. Test method='quantile'

        # Get the correct result with remove_first=False
        values_res = np.array([[0, 3, 0, 3],
                               [2, 0, 0, 2],
                               [3, 0, 1, 2],
                               [2, 2, 1, 1],
                               [1, 0, 0, 0],
                               [2, 1, 0, 0],
                               [3, 2, 1, 4],
                               [0, 3, 0, 0],
                               [0, 1, 1, 1],
                               [1, 2, 0, 0]])
        columns_res = ['c#0', 'c#1', 'c#2', 'c#3', 'a#0', 'a#1', 'a#2', 'a#3',
                       'd', 'b#0', 'b#1', 'b#2', 'b#3', 'b#4']
        enc = OneHotEncoder(sparse=True)
        X_bin_res = enc.fit_transform(values_res)

        # Create the FeatureBinarizer
        n_cuts = 3
        for get_type in ["auto", "column_names"]:
            binarizer = FeaturesBinarizer(method='quantile', n_cuts=n_cuts,
                                          get_type=get_type,
                                          remove_first=False)

            # Apply it on the features matrix
            features_bin = binarizer.fit(features)
            X_bin = features_bin.transform(features)
            X_bin_fit_transform = binarizer.fit_transform(features)

            self.assertTrue((X_bin != X_bin_res).nnz == 0)
            self.assertTrue((X_bin_fit_transform != X_bin_res).nnz == 0)
            np.testing.assert_equal(columns_res, features_bin._columns_names)
            self.assertTrue(binarizer.blocks_start == [0, 4])
            self.assertTrue(binarizer.blocks_length == [4, 4])
            np.testing.assert_equal(
                np.around(binarizer.bins_boundaries['a:continuous'], 4),
                np.array([-np.inf, -0.7188, -0.1915, 0.4458, np.inf]))
            np.testing.assert_equal(
                np.around(binarizer.bins_boundaries['c:continuous'], 4),
                np.array([-np.inf, 0.009, 0.2711, 0.4732, np.inf]))

        # Get the correct result with remove_first=True
        values_res = np.array([[0, 2, 0, 2],
                               [1, 0, 0, 1],
                               [2, 0, 1, 1],
                               [1, 1, 1, 0],
                               [0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [2, 1, 1, 3],
                               [0, 2, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0]])
        columns_res = ['c#0', 'c#1', 'c#2', 'a#0', 'a#1', 'a#2', 'd', 'b#0',
                       'b#1', 'b#2', 'b#3']
        enc = OneHotEncoder(sparse=True)
        X_bin_res = enc.fit_transform(values_res)

        # Create the FeatureBinarizer
        n_cuts = 3
        for get_type in ["auto", "column_names"]:
            binarizer = FeaturesBinarizer(method='quantile', n_cuts=n_cuts,
                                          get_type=get_type,
                                          remove_first=True)

            # Apply it on the features matrix
            features1, features2 = features.copy(), features.copy()
            features_bin = binarizer.fit(features1)
            X_bin = features_bin.transform(features1)
            X_bin_fit_transform = binarizer.fit_transform(features2)
            self.assertTrue((X_bin_fit_transform != X_bin_res).nnz == 0)
            self.assertTrue((X_bin != X_bin_res).nnz == 0)
            np.testing.assert_equal(columns_res,
                                    features_bin._columns_names)

        # 2. Test method='linspace'

        # Get the correct result with remove_first=False
        values_res = np.array([[1, 2, 0, 3],
                               [1, 0, 0, 2],
                               [1, 0, 1, 2],
                               [1, 1, 1, 1],
                               [1, 0, 0, 0],
                               [1, 0, 0, 0],
                               [3, 1, 1, 4],
                               [0, 3, 0, 0],
                               [0, 1, 1, 1],
                               [1, 2, 0, 0]])
        columns_res = ['c#0', 'c#1', 'c#2', 'c#3', 'a#0', 'a#1', 'a#2', 'a#3',
                       'd', 'b#0', 'b#1', 'b#2', 'b#3', 'b#4']
        enc = OneHotEncoder(sparse=True)
        X_bin_res = enc.fit_transform(values_res)

        # Create the FeatureBinarizer
        n_cuts = 3
        for get_type in ["auto", "column_names"]:
            binarizer = FeaturesBinarizer(method='linspace', n_cuts=n_cuts,
                                          get_type=get_type,
                                          remove_first=False)

            # Apply it on the features matrix
            features_bin = binarizer.fit(features)
            X_bin = features_bin.transform(features)
            X_bin_fit_transform = binarizer.fit_transform(features)

            self.assertTrue((X_bin != X_bin_res).nnz == 0)
            self.assertTrue((X_bin_fit_transform != X_bin_res).nnz == 0)
            np.testing.assert_equal(columns_res,
                                    features_bin._columns_names)
            self.assertTrue(binarizer.blocks_start == [0, 4])
            self.assertTrue(binarizer.blocks_length == [4, 4])
            np.testing.assert_equal(
                np.around(binarizer.bins_boundaries['a:continuous'], 4),
                np.array([-np.inf, -0.4952, 0.1242, 0.7435, np.inf]))
            np.testing.assert_equal(
                np.around(binarizer.bins_boundaries['c:continuous'], 4),
                np.array([-np.inf, -0.231, 0.5272, 1.2853, np.inf]))

        # Get the correct result with remove_first=True
        values_res = np.array([[0, 1, 0, 2],
                               [0, 0, 0, 1],
                               [0, 0, 1, 1],
                               [0, 0, 1, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [2, 0, 1, 3],
                               [0, 2, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0]])
        columns_res = ['c#0', 'c#1', 'c#2', 'a#0', 'a#1', 'a#2', 'd', 'b#0',
                       'b#1', 'b#2', 'b#3']
        enc = OneHotEncoder(sparse=True)
        X_bin_res = enc.fit_transform(values_res)

        # Create the FeatureBinarizer
        n_cuts = 3
        for get_type in ["auto", "column_names"]:
            binarizer = FeaturesBinarizer(method='linspace', n_cuts=n_cuts,
                                          get_type=get_type,
                                          remove_first=True)

            # Apply it on the features matrix
            features1, features2 = features.copy(), features.copy()
            features_bin = binarizer.fit(features1)
            X_bin = features_bin.transform(features1)

            X_bin_fit_transform = binarizer.fit_transform(features2)

            self.assertTrue((X_bin_fit_transform != X_bin_res).nnz == 0)
            self.assertTrue((X_bin != X_bin_res).nnz == 0)
            np.testing.assert_equal(columns_res,
                                    features_bin._columns_names)
        return

if __name__ == "main":
    unittest.main()

import unittest
import numpy as np
import pandas as pd

from features_binarizer import FeaturesBinarizer


class Test(unittest.TestCase):

    def test_FeaturesBinarizer(self):
        """... Test FeaturesBinarizer
        """

        # TODO: unit test for fit and then fit_transform
        # TODO: unit test for method="linspace"

        # Create a features matrix
        values = np.array([[0.00902084,  0.54159776, 0., 3.],
                           [0.46599565, -0.71875887, 0., 2.],
                           [0.52091721, -0.83803094, 1., 2.],
                           [0.47315496,  0.0730993, 1., 1.],
                           [0.08180209, -1.11447889, 0., 0.],
                           [0.45011727, -0.57931684, 0., 0.],
                           [2.04347947, -0.10127498, 1., 7.],
                           [-0.98909384,  1.36281079, 0., 0.],
                           [-0.30637613, -0.19147753, 1., 1.],
                           [0.27110903,  0.44583304, 0., 0.]])
        values = values[:, :2]
        columns = ['c:continuous', 'a:continuous']#, 'd', 'b:discrete']
        features = pd.DataFrame(values, columns=columns)

        # Create the FeatureBinarizer
        n_cuts = 3
        binarizer = FeaturesBinarizer(method='quantile', n_cuts=n_cuts,
                                      remove_first=False)

        print(binarizer)

        # Apply it on the features matrix
        features_bin = binarizer.fit(features)
        print(features_bin.transform(features))
        print(features_bin.enc.feature_indices_)

        return
        # features_bin = binarizer.fit_transform(features)

        # Get the correct result
        values_res = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                               [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                               [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                               [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                               [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                               [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                               [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]])

        columns_res = ['c#0', 'c#1', 'c#2', 'a#0', 'a#1', 'a#2', 'd',
                       'b#0', 'b#1', 'b#2', 'b#3', 'b#4']

        np.testing.assert_equal(features_bin.values, values_res)
        np.testing.assert_equal(columns_res,
                                features_bin.columns.values)
        self.assertTrue(binarizer.blocks_start == [0, 3])
        self.assertTrue(binarizer.blocks_length == [3, 3])

        np.testing.assert_equal(
                binarizer.bins_boundaries['a:continuous'],
                np.array([-np.inf, -0.57931684, 0.0730993, np.inf]))

        np.testing.assert_equal(
                binarizer.bins_boundaries['c:continuous'],
                np.array([-np.inf, 0.08180209, 0.46599565, np.inf]))

if __name__ == "main":
    unittest.main()

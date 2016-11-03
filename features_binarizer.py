import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


# TODO: put back the method="linspace" case


class FeaturesBinarizer(BaseEstimator, TransformerMixin):
    """
    This is a scikit-learn transformer that transform an input
    pandas DataFrame X of shape (n_samples, n_features) into a binary
    matrix of size (n_samples, n_new_features).
    Continous features (columns with name ending with ":continuous")
    are modified and extended into binary features, using linearly or
    inter-quantiles spaced bins.
    Discrete features (columns with name ending with ":discrete") are
    binary encoded with K columns, where K is the number of modalities.
    Other features (none of the above) are left unchanged.
    Parameters
    ----------
    n_cuts : `int`, default=10
        Number of cuts
    method : "quantile" or "linspace", default="quantile"
        * If ``"quantile"`` quantile-based cuts are used
        * If ``"linspace"`` linearly spaced cuts are used
    Attributes
    ----------
    bins_boundaries : `list`
    blocks_start : ``
    blocks_length : ``
    References
    ----------
    http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
    """

    _attrinfos = {
        "method": {
            "writable": False
        },
        "n_cuts": {
            "writable": False
        },
        "remove_first": {
            "writable": False
        },
        "_prb": {
            "writable": False
        },
        "_encoders": {
            "writable": True
        },
        "bins_boundaries": {
            "writable": False
        },
        "blocks_start": {
            "writable": False
        },
        "blocks_length": {
            "writable": False
        },
        "_columns_names": {
            "writable": False
        },
        "_idx_col": {
            "writable": False
        }
    }

    def __init__(self, method="quantile", n_cuts= 10, remove_first= True):
        self.method = method
        self.n_cuts = n_cuts
        self.remove_first = remove_first
        self._init()

    def _init(self):
        # Quantile probabilities
        self._prb = np.linspace(0, 100, self.n_cuts + 1)
        # OneHotEncoders for continuous and discrete features
        self._encoders = {}
        # Bins boundaries for continuous features
        self.bins_boundaries = {}
        # The first column of continuous features (will be
        #  useful for ProxBinarsity)
        self.blocks_start = []
        # Number of columns for each continuous features (will be
        #  useful for ProxBinarsity)
        self.blocks_length = []
        self._columns_names = {}
        self._idx_col = 0

    # def _fit_linspace(self, X, y=None):
    #     """
    #     Compute linearly spaced cuts of each features
    #     """
    #     # Maximum and minimum of each feature
    #     X_max = np.max(X, axis=0)
    #     X_min = np.min(X, axis=0)
    #     # compute the cuts at each feature
    #     self.bins_ = [np.linspace(X_min[i], X_max[i], self.n_cuts)
    # for i in range(X.shape[1])]
    #     return self

    # def _transform_linspace(self, X):
    #     X = np.array(X)
    #     X_new_cols = []
    #     mask = self.mask_
    #     tv_mask = []
    #     for i in range(X.shape[1]):
    #         if mask[i]:
    #             # distance to all cuts
    #             tmp = np.abs(X[:, i][:, None] - self.bins_[i])
    #             # give it the label of the closest cut
    #             bin_cols = (np.argmin(tmp, axis=1)[:, None] == np.arange(self.n_cuts))
    #             X_new_cols.append(bin_cols.astype(np.float))
    #             tv_mask.extend([True] * bin_cols.shape[1])
    #         else:
    #             X_new_cols.append(X[:, i][:, None])
    #             tv_mask.append(False)
    #     self.tv_mask = tv_mask
    #     return np.concatenate(X_new_cols, axis=1)

    def _fit_quantile(self, X, y=None):
        self._init()
        for feat_name in X:
            feat = X[feat_name]
            _ = self._binarize(feat_name, feat, True)
        return self

    def _transform_quantile(self, X):
        x_new = pd.DataFrame()
        for feat_name in X:
            feat = X[feat_name]
            feat = self._binarize(feat_name, feat, False)
            x_new = pd.concat([x_new, feat], axis=1)
        return x_new

    def _fit_transform_quantile(self, X):
        self._init()
        x_new = pd.DataFrame()
        for feat_name in X:
            feat = X[feat_name]
            feat = self._binarize(feat_name, feat, True)
            x_new = pd.concat([x_new, feat], axis=1)
        return x_new

    def _binarize(self, feat_name, feat, fit):
        """
        Binarize a single feature
        Parameters
        ----------
        feat_name : `str`
            The feature name
        feat : `np.array`, shape=(n_samples,)
            The column containing the feature to be binarized
        fit : `bool`
            If `True`, we need to fit (compute quantiles) for this
            feature
        Returns
        -------
        output : `np.ndarray`, shape=(n_samples, ?)
            The binarized feature. The number of columns is smaller or
            equal to ``n_cuts``, depending on the ``method`` and/or on
            the actual number of distinct quantiles for this feature
        """
        continuous = feat_name.endswith(":continuous")
        discrete = feat_name.endswith(":discrete")
        idx_col = self._idx_col
        encoders = self._encoders

        if not continuous and not discrete:
            idx_col += 1
        else:
            feat_type = ":discrete"
            if continuous:
                feat_type = ":continuous"
                # Compute quantiles for the feature
                quantiles = self._get_quantiles(feat_name, feat, fit)
                # Discretize feature
                feat = pd.cut(feat, quantiles, labels=False)

            n_samples = feat.shape[0]
            if fit:
                encoder = OneHotEncoder(dtype=np.int, sparse=True)
                # Binarize feature
                feat = encoder.fit_transform(feat.reshape(n_samples, 1))
                # Save the encoder
                encoders[feat_name] = encoder
            else:
                encoder = encoders[feat_name]
                feat = encoder.transform(feat.reshape(n_samples, 1))

            if self.remove_first:
                feat = feat[:, 1:]

            n_cols_feat = feat.shape[1]

            if continuous:
                self.blocks_start.append(idx_col)
                self.blocks_length.append(n_cols_feat)

            idx_col += n_cols_feat

            columns = self._get_columns_names(feat_name,
                                              feat_type,
                                              n_cols_feat,
                                              fit)

            feat = pd.DataFrame(feat, columns=columns)
        self._idx_col = idx_col
        return feat

    def _get_quantiles(self, feat_name, x, fit):
        if fit:
            q = np.percentile(x, self._prb, interpolation="nearest")
            q = np.unique(q)
            q[0] = -np.inf
            q[-1] = np.inf
            self.bins_boundaries[feat_name] = q
        else:
            q = self.bins_boundaries[feat_name]
        return q

    def _get_columns_names(self, feat_name, feat_type, n_bins, fit):
        if fit:
            if self.remove_first:
                columns = [feat_name.replace(feat_type, "") + "#" \
                            + str(i) for i in range(1, n_bins + 1)]
            else:
                columns = [feat_name.replace(feat_type, "") + "#" \
                            + str(i) for i in range(n_bins)]
            self._columns_names[feat_name] = columns
        else:
            columns = self._columns_names[feat_name]
        return columns

    def fit(self, X, y=None):
        """
        Fit the binarization using the features matrix
        Parameters
        ----------
        X : `pd.DataFrame`, shape=(n_samples, n_features)
            The features matrix
        Returns
        -------
        output : `FeaturesBinarizer`
            The fitted current instance
        """
        if self.method == 'quantile':
            return self._fit_quantile(X, y)
        elif self.method == 'linspace':
            raise ValueError("Method %s not implemented" % self.method)
        else:
            raise ValueError("Method %s not implemented" % self.method)

    def transform(self, X):
        """
        Apply the binarization to the given features matrix
        Parameters
        ----------
        X : `pd.DataFrame`, shape=(n_samples, n_features)
            The features matrix
        Returns
        -------
        output : `pd.DataFrame`
            The binarized features matrix. The number of columns is
            larger than n_features, smaller than n_cuts * n_features,
            depending on the actual number of columns that have been
            binarized
        """
        if self.method == 'quantile':
            return self._transform_quantile(X)
        elif self.method == 'linspace':
            raise ValueError("Method %s not implemented" % self.method)
        else:
            raise ValueError("Method %s not implemented" % self.method)

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit and apply the binarization using the features matrix
        Parameters
        ----------
        X : `pd.DataFrame`, shape=(n_samples, n_features)
            The features matrix
        Returns
        -------
        output : `pd.DataFrame`
            The binarized features matrix. The number of columns is
            larger than n_features, smaller than n_cuts * n_features,
            depending on the actual number of columns that have been
            binarized
        """
        if self.method == 'quantile':
            return self._fit_transform_quantile(X)
        elif self.method == 'linspace':
            raise ValueError("Method %s not implemented" % self.method)
        else:
            raise ValueError("Method %s not implemented" % self.method)
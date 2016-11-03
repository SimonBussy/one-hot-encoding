import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


# TODO: add discrete features 


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

    def _fit_quantile(self, X, y=None):
        quantilized_X = pd.DataFrame()
        self._init()
        for feat_name in X:
            feat = X[feat_name]
            quantalized_feat = self._quantilize(feat_name, feat, fit=True)
            quantilized_X[feat_name] = quantalized_feat

        self.enc = OneHotEncoder(sparse=True)
        self.enc.fit(quantilized_X)

        return self

    def _transform_quantile(self, X):
        """Apply the binarization using the features matrix
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
        quantilized_X = pd.DataFrame()
        for feat_name in X:
            feat = X[feat_name]
            quantilized_feat = self._quantilize(feat_name, feat, fit=False)
            quantilized_X[feat_name] = quantilized_feat

        return self.enc.transform(quantilized_X)

    def _quantilize(self, feat_name, feat, fit=False):
        """Binarize a single feature
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
        # Compute quantiles for the feature
        quantiles = self._get_quantiles(feat_name, feat, fit)
        # Discretize feature
        feat = pd.cut(feat, quantiles, labels=False)

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

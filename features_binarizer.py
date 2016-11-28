import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class FeaturesBinarizer(BaseEstimator, TransformerMixin):
    """This is a scikit-learn transformer that transform an input
    pandas DataFrame X of shape (n_samples, n_features) into a binary
    matrix of size (n_samples, n_new_features).
    Continous features are modified and extended into binary features, using
    linearly or inter-quantiles spaced bins.
    Discrete features are binary encoded with K columns, where K is the number
    of modalities.
    Other features (none of the above) are left unchanged.
    Parameters
    ----------
    n_cuts : `int`, default=10
        Number of cuts
    method : "quantile" or "linspace", default="quantile"
        * If ``"quantile"`` quantile-based cuts are used.
        * If ``"linspace"`` linearly spaced cuts are used.
    Attributes
    ----------
    bins_boundaries : `list`
    blocks_start : ``
    blocks_length : ``
    References
    ----------
    http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
    """

    def __init__(self, method="quantile", n_cuts=10, get_type="auto",
                 remove_first=False):
        self.method = method
        self.n_cuts = n_cuts
        self.get_type = get_type
        self.remove_first = remove_first
        self._init()

    def _init(self):
        # Quantile probabilities
        self._prb = np.linspace(0, 100, self.n_cuts + 2)
        # OneHotEncoders for continuous and discrete features
        self.enc = OneHotEncoder(sparse=True)
        # Bins boundaries for continuous features
        self.bins_boundaries = {}
        # Features type
        self.feature_type = {}
        # The first column of continuous features (will be
        #  useful for ProxBinarsity)
        self.blocks_start = []
        # Number of columns for each continuous features (will be
        #  useful for ProxBinarsity)
        self.blocks_length = []
        self._columns_names = []
        self._idx_col = 0

    def fit(self, X, y=None):
        """Fit the binarization using the features matrix.
        Parameters
        ----------
        X : `pd.DataFrame`, shape=(n_samples, n_features)
            The features matrix.
        Returns
        -------
        output : `FeaturesBinarizer`
            The fitted current instance.
        """
        self._init()
        binarized_X = pd.DataFrame()
        for feat_name in X:
            feat = X[feat_name]
            binarized_feat = self._binarize(feat_name, feat, fit=True)
            binarized_X[feat_name] = binarized_feat
        self.enc.fit(binarized_X)
        return self

    def transform(self, X):
        """Apply the binarization to the given features matrix.
        Parameters
        ----------
        X : `pd.DataFrame`, shape=(n_samples, n_features)
            The features matrix.
        Returns
        -------
        output : `pd.DataFrame`
            The binarized features matrix. The number of columns is
            larger than n_features, smaller than n_cuts * n_features,
            depending on the actual number of columns that have been
            binarized.
        """
        binarized_X = pd.DataFrame()
        for feat_name in X:
            feat = X[feat_name]
            binarized_feat = self._binarize(feat_name, feat, fit=False)
            binarized_X[feat_name] = binarized_feat
        return self.enc.transform(binarized_X)

    def fit_transform(self, X, y=None, **kwargs):
        """Fit and apply the binarization using the features matrix.
        Parameters
        ----------
        X : `pd.DataFrame`, shape=(n_samples, n_features)
            The features matrix.
        Returns
        -------
        output : `pd.DataFrame`
            The binarized features matrix. The number of columns is
            larger than n_features, smaller than n_cuts * n_features,
            depending on the actual number of columns that have been
            binarized.
        """
        self._init()
        binarized_X = pd.DataFrame()
        for feat_name in X:
            feat = X[feat_name]
            binarized_feat = self._binarize(feat_name, feat, fit=True)
            binarized_X[feat_name] = binarized_feat
        return self.enc.fit_transform(binarized_X)

    def _binarize(self, feat_name, feat, fit=False):
        """Binarize a single feature.
        Parameters
        ----------
        feat_name : `str`
            The feature name.
        feat : `np.array`, shape=(n_samples,)
            The column containing the feature to be binarized.
        fit : `bool`
            If `True`, we need to fit (compute boundaries) for this feature.
        Returns
        -------
        output : `np.ndarray`, shape=(n_samples, ?)
            The binarized feature. The number of columns is smaller or
            equal to ``n_cuts``, depending on the ``method`` and/or on
            the actual number of distinct boundaries for this feature.
        """
        feat_type = self._get_feature_type(feat_name, feat, fit)
        discrete = feat_type == "discrete"
        continuous = feat_type == "continuous"
        idx_col = self._idx_col
        if continuous or discrete:
            if continuous:
                # Compute bins boundaries for the feature
                boundaries = self._get_boundaries(feat_name, feat, fit)
                # Discretize feature
                feat = pd.cut(feat, boundaries, labels=False)
                n_cols_feat = len(boundaries) - 1
                self.blocks_start.append(idx_col)
                self.blocks_length.append(n_cols_feat)
            else:
                n_cols_feat = len(feat.unique())
            if self.remove_first:
                n_cols_feat -= 1
                if continuous or fit:
                    feat[feat > 0] -= 1
            if fit:
                columns_names = self._get_columns_names(feat_name, feat_type,
                                                        n_cols_feat)
                self._columns_names += columns_names
            idx_col += n_cols_feat
        else:
            idx_col += 1
            self._columns_names.append(feat_name)
        self._idx_col = idx_col
        return feat

    def _get_feature_type(self, feat_name, feat, fit=False):
        """Get the type of a single feature.
        Parameters
        ----------
        feat_name : `str`
            The feature name.
        feat : `np.array`, shape=(n_samples,)
            The column containing the feature to be binarized.
        fit : `bool`
            If `True`, we need to fit (compute boundaries) for this feature.
        Returns
        -------
        output : `str`
            The type of the feature. If self.get_type is "column_names", columns
            with name ending with ":continuous" means continous features,
            columns with name ending with ":discrete" means discrete features,
            and other features (none of the above) are left unchanged. If
            self.get_type is "auto", an automatic type detection procedure is
            followed.
        """
        if fit:
            if self.get_type == "column_names":
                if feat_name.endswith(":continuous"):
                    feat_type = "continuous"
                elif feat_name.endswith(":discrete"):
                    feat_type = "discrete"
                else:
                    feat_type = ""
            elif self.get_type == "auto":
                # threshold choice depending on whether one has more than 20
                # examples or not
                if len(feat) > 30:
                    eps = 15
                else:
                    eps = len(feat) / 2
                # count distinct realizations and compare to threshold
                if len(feat.unique()) > eps:
                    feat_type = "continuous"
                elif len(feat.unique()) > 2:
                    feat_type = "discrete"
                else:
                    feat_type = ""
            else:
                raise ValueError(
                    "get_type %s not implemented" % self.get_type)
            self.feature_type[feat_name] = feat_type
        else:
            feat_type = self.feature_type[feat_name]
        return feat_type

    def _get_boundaries(self, feat_name, feat, fit):
        """Get bins boundaries of a single feature.
        Parameters
        ----------
        feat_name : `str`
            The feature name.
        feat : `np.array`, shape=(n_samples,)
            The column containing the feature to be binarized.
        fit : `bool`
            If `True`, we need to fit (compute boundaries) for this feature.
        Returns
        -------
        output : `np.ndarray`, shape=(?,)
            The bins boundaries. The number of lines is smaller or
            equal to ``n_cuts``, depending on the ``method`` and/or on
            the actual number of distinct boundaries for this feature.
        """
        if fit:
            if self.method == 'quantile':
                boundaries = np.percentile(feat, self._prb,
                                           interpolation="nearest")
                # Only keep distinct bins boundaries
                boundaries = np.unique(boundaries)
            elif self.method == 'linspace':
                # Maximum and minimum of the feature
                feat_max = np.max(feat)
                feat_min = np.min(feat)
                # Compute the cuts
                boundaries = np.linspace(feat_min, feat_max, self.n_cuts + 2)
            else:
                raise ValueError("Method %s not implemented" % self.method)
            boundaries[0] = -np.inf
            boundaries[-1] = np.inf
            self.bins_boundaries[feat_name] = boundaries
        else:
            boundaries = self.bins_boundaries[feat_name]
        return boundaries

    @staticmethod
    def _get_columns_names(feat_name, feat_type, n_bins):
        columns = [feat_name.replace(':' + feat_type, "") + "#" \
                   + str(i) for i in range(n_bins)]
        return columns

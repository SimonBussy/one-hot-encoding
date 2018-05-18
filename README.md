# one-hot-encoding

This is an initial work to create a scikit-learn transformer that transforms an input pandas DataFrame X of shape (n_samples, n_features) 
into a binary matrix of size (n_samples, n_new_features).
Continous features are modified and extended into binary features, using linearly or inter-quantiles spaced bins.
Discrete features are binary encoded with K columns, where K is the number of modalities.
Other features (none of the above) are left unchanged.

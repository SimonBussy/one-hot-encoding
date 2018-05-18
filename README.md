# one-hot-encoding

This is a preliminary work to produce a scikit-learn transformer that transforms an input matrix of shape (n_samples, n_features) into a binary matrix of size (n_samples, n_new_features).
Continous features are modified and extended into binary features, using linearly or inter-quantiles spaced bins.
Discrete features are binary encoded with K columns, where K is the number of modalities.
Other features (none of the above) are left unchanged.

This work have been updated and integrated to the [tick](https://x-datainitiative.github.io/tick/) module as a preprocessing tool ([here](https://github.com/X-DataInitiative/tick/blob/master/tick/preprocessing/features_binarizer.py)) and used in the paper "Binarsity: a penalization for one-hot encoded features" available [here](https://arxiv.org/abs/1703.08619).

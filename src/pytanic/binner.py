import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict

# Class for binning
class Binner(BaseEstimator, TransformerMixin):
    def __init__(self,
                 features,
                 bin_dict: Dict = {'Age': [-np.inf, 0, 17, 22, 27, 31, 36, 46, 55, np.inf],
                                   'Fare': [-np.inf, 7.7, 8.1, 12.5, 19.3, 28, 57, np.inf]}):
        self.features = features
        self.bin_dict = bin_dict
        for feature in self.features:
            assert feature in self.bin_dict, f'Feature {feature} is missing in bin_dict'

    def fit(self, X, y=None) -> None:
        return self

    def transform(self, X):
        X_new = X.copy()
        for feature in self.features:
            X_new[f'{feature}_bin'] = pd.cut(X_new[feature].fillna(-1),
                                                     bins=self.bin_dict[feature],
                                                     labels=False)
        return X_new
import pandas as pd
from pandas import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict

# Class for OHE
class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 features,
                 feature_values: Dict = {'Pclass': [1, 2, 3],
                                         'Sex': ['male', 'female'],
                                         'Embarked': ['S', 'C', 'Q']},
                 drop_first: bool = True):
        self.features = features
        self.feature_values = feature_values
        self.drop_first = drop_first
        for feature in self.features:
            assert feature in self.feature_values, f'Feature {feature} is missing in feature_values'

    def fit(self, X, y=None) -> None:
        return self

    def transform(self, X):
        X_new = X.copy()
        for feature in self.features:
            X_new[feature] = X_new[feature].astype(
                CategoricalDtype(categories=self.feature_values[feature], ordered=True)
            )
            X_new = X_new.join(pd.get_dummies(X_new[feature], prefix=feature, drop_first=self.drop_first))
        return X_new
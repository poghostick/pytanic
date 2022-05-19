import numpy as np
import pandas as pd
from pandas import CategoricalDtype
import re
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

# Class for additional features extracted from data
class AttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, features, drop_first: bool = True):
        self.features = features
        self.drop_first = drop_first

    def fit(self, X, y=None) -> None:
        self.family_survival_rate = self.build_survival_rate(X, y, col_name='Name')
        self.ticket_survival_rate = self.build_survival_rate(X, y, col_name='Ticket')
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_new = X.copy()
        deck_cat = CategoricalDtype(categories=['ABC', 'DE', 'FG', 'M'], ordered=True)
        # X_new['Deck'] = X_new['Cabin'].apply(self.get_deck, args=(['ABC', 'DE', 'FG', 'M'])).astype(deck_cat)
        X_new['Deck'] = X_new['Cabin'].apply(lambda row: self.get_deck(row, ['ABC', 'DE', 'FG', 'M'])).astype(deck_cat)
        family_size_cat = CategoricalDtype(categories=['Alone', 'Small', 'Medium', 'Large'], ordered=True)
        X_new['FamilySize'] = X_new.apply(self.get_family_size, axis=1).astype(family_size_cat)
        title_cat = CategoricalDtype(categories=['mr', 'miss/mrs/ms', 'dr/military/noble/clergy'])
        X_new['title'] = self.get_title(X_new['Name']).astype(title_cat)
        for feature in ['FamilySize', 'title']:
            X_new = X_new.join(pd.get_dummies(X_new[feature], prefix=feature, drop_first=self.drop_first))
        X_new['is_married'] = 0
        X_new.loc[X_new['Name'].str.contains('Mrs.'), 'is_married'] = 1
        X_new['family'] = self.get_families(X_new['Name'])
        X_new = X_new.merge(self.family_survival_rate, how='left', left_on='family', right_index=True)
        X_new = X_new.merge(self.ticket_survival_rate, how='left', left_on='Ticket', right_index=True)
        for feature in ['family', 'ticket']:
            X_new[f'{feature}_survival_rate_na'] = np.where(X_new[f'{feature}_freq'].isna(), 1, 0)
            feature_cols = [f'{feature}_freq', f'{feature}_surv_rate']
            X_new.loc[:, feature_cols] = X_new.loc[:, feature_cols].fillna(0)
        columns = X_new.columns.difference(['PassengerId', 'Name', 'Pclass', 'Sex', 'Age', 'Ticket', 'title',
                                           'Fare', 'Cabin', 'Embarked', 'Deck', 'FamilySize', 'family'])

        return X_new[columns]

    @staticmethod
    def get_deck(s: str, values: List) -> str:
        if s is np.nan:
            return values[-1]
        else:
            for deck in values[:-1]:
                if s[0] in deck:
                    return deck
        return values[-1]

    @staticmethod
    def get_family_size(df: pd.DataFrame):
        family_size = 1 + df['SibSp'] + df['Parch']
        if family_size <= 1:
            return 'Alone'
        elif family_size <= 4:
            return 'Small'
        elif family_size <= 6:
            return 'Medium'
        else:
            return 'Large'

    @staticmethod
    def build_survival_rate(X: pd.DataFrame, y: pd.Series, col_name: str) -> pd.DataFrame:
        def extract_surname(name: str):
            return re.search(r'([a-z]+),', name, flags=re.IGNORECASE).group(1).lower()
        survival_rate = X.copy()
        survival_rate['Survived'] = y
        if col_name == 'Name':
            survival_rate['Family'] = survival_rate['Name'].apply(extract_surname)
            survival_rate = survival_rate.groupby('Family').agg(
                family_freq=pd.NamedAgg(column='Survived', aggfunc='count'),
                family_surv_rate=pd.NamedAgg(column='Survived', aggfunc='mean')
            )
            return survival_rate.query('family_freq > 1')
        else:
            survival_rate = survival_rate.groupby('Ticket').agg(
                ticket_freq=pd.NamedAgg(column='Survived', aggfunc='count'),
                ticket_surv_rate=pd.NamedAgg(column='Survived', aggfunc='mean')
            )
            return survival_rate.query('ticket_freq > 1')

    @staticmethod
    def get_title(names: pd.Series):
        def find_title(name: str):
            title = re.search(r'([a-z]+),\s*([a-z]+)\.?', name, flags=re.IGNORECASE).group(2).lower()
            if title in ('miss', 'mrs', 'ms', 'mlle', 'lady', 'mme', 'the Countess', 'dona'):
                return 'miss/mrs/ms'
            elif title in ('dr', 'col', 'major', 'master', 'jonkheer', 'capt', 'sir', 'don', 'rev'):
                return 'dr/military/noble/clergy'
            else:
                return title
        return names.apply(find_title)

    @staticmethod
    def get_families(names: pd.Series):
        def extract_surname(name: str):
            return re.search(r'([a-z]+),', name, flags=re.IGNORECASE).group(1).lower()
        return names.apply(extract_surname)
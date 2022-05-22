import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Dict, List, Tuple
import zipfile

from pytanic.binner import Binner
from pytanic.encoder import Encoder
from pytanic.attribute_adder import AttributeAdder

def load_data() -> Tuple[pd.DataFrame]:
    """
    Unzip, load and return training and test sets.
    """
    data_path = Path().absolute() / 'data'
    if not Path.is_file(data_path / 'train.csv') or not Path.is_file(data_path / 'test.csv'):
        with zipfile.ZipFile(data_path / 'titanic.zip', 'r') as zip_ref:
            zip_ref.extractall(data_path)
    train_df = pd.read_csv(data_path / 'train.csv')
    test_df = pd.read_csv(data_path / 'test.csv')
    return train_df, test_df


train_df, test_df = load_data()

pipe = Pipeline([
    ('num_imputer', Binner(['Age', 'Fare'])),
    ('encoder', Encoder(['Pclass', 'Sex', 'Embarked'])),
    ('attribute_adder', AttributeAdder(['Cabin', 'SibSp', 'Parch', 'Name', 'Ticket'])),
    ('model', LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen'))
])

pipe.fit(train_df.drop('Survived', axis=1), train_df['Survived'])

train_df_mod = pipe.fit_transform(train_df.drop('Survived', axis=1), train_df['Survived'])
test_df_mod = pipe.transform(test_df)
from pyrsistent import PClass, field, pvector_field
import pandas as pd
import numpy as np
from typing import List

from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper
from sklearn.datasets import make_classification, fetch_covtype


def build_generic_df_mapper(df, scalar_pipe=None) -> DataFrameMapper:
    if scalar_pipe is None:
        scalar_pipe = lambda: MissingIndicator()

    df_cols = []
    dropped_columns = set()
    for c in df.columns:
        if np.all(pd.isnull(df[c])):
            continue
        if df[c].nunique(dropna=False) <= 1:
            continue

        if df[c].dtype in ['float64', 'int', 'bool']:
            if df[c].isnull().any():
                df_cols.append(([c], scalar_pipe()))
            else:
                df_cols.append(([c], None))
        elif df[c].dtype == 'O':
            df_cols.append(([c], OneHotEncoder()))
        else:
            dropped_columns |= set([c])

    return DataFrameMapper(df_cols)


class ModelScenario(PClass):
    data_path = field(str)
    data_source_name = field(str)
    data_source_id_col = field(str, mandatory=False)  # drop this column
    data_source_outcome_col = field(str)

    @property
    def dataset_shifted(self) -> bool:
        return False

    @property
    def data_source(self):
        if hasattr(self, 'data_source_description'):
            return self.data_source_description
        else:
            return self.data_source_name

    def load_data(self):
        df = pd.read_csv(self.data_path)
        for c in df.columns:
            if df[c].dtype == 'bool':
                df[c] = df[c].astype(float)
        if hasattr(self, 'data_source_id_col'):
            df.drop(columns=[self.data_source_id_col], inplace=True)

        return self.preprocess_data(df)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def base_sample_weight(self, X, y):
        if isinstance(y, np.ndarray):
            return np.ones(shape=y.shape)
        elif isinstance(y, pd.Series):
            return np.ones(shape=(len(y),))
        else:
            raise ValueError("y must be an np.array or pd.series")

    def train_test(self, legacy_size=0.7, legacy_drop_frac=0.5, legacy_missing_label=0.2):
        X = self.load_data()

        assert 'legacy' not in X.columns
        X['legacy'] = bernoulli(legacy_size).rvs(len(X))

        y = X[self.data_source_outcome_col]
        X = X.drop(columns=[self.data_source_outcome_col])

        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75)

        train_y -= train_X['legacy'] * train_y * bernoulli(legacy_missing_label).rvs(len(train_y))

        for c in train_X.columns:
            if (bernoulli(legacy_drop_frac).rvs() == 1) and (c != 'legacy'):
                train_X.loc[train_X['legacy'] == 1, c] = None

        return train_X, test_X, train_y, test_y

    def build_pipeline(self, train_X: pd.DataFrame) -> Pipeline:
        return Pipeline([
            ('transform', build_generic_df_mapper(train_X)),
            ('final_estimator', HistGradientBoostingClassifier()),
        ])


class _CarInsuranceModelScenario(ModelScenario):
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in df.columns:
            if set(df[c].unique()) == set(['Yes', 'No']):
                df[c] = (df[c] == 'Yes')
        for c in df.columns:
            if df[c].dtype == 'bool':
                df[c] = df[c].astype(float)
        return df

class _TabularPlaygroundModelScenario(ModelScenario):
    def build_pipeline(self, train_X: pd.DataFrame) -> Pipeline:
        return Pipeline([
            ('transform', build_generic_df_mapper(train_X, scalar_pipe=lambda: [SimpleImputer(), StandardScaler()])),
            ('final_estimator', HistGradientBoostingClassifier()),
        ])


class SyntheticScenario(ModelScenario):
    data_source_outcome_col = field(str, initial='outcome')

    n_samples = field(int, initial=100000)
    n_features = field(int, initial=100)
    n_informative = field(int, initial=20)
    n_redundant = field(int, initial=10)
    n_repeated = field(int, initial=5)
    n_clusters_per_class = field(int, initial=4)
    weights = pvector_field(float, initial=[0.95, 0.05])

    @property
    def data_source_description(self):
        return 'sklearn.datasets.make_classification'

    def load_data(self):
        X, y = make_classification(
            n_samples=100000,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_repeated=self.n_repeated,
            n_clusters_per_class=self.n_clusters_per_class,
            weights=list(self.weights),
        )

        df = pd.DataFrame(X)
        df.columns=['feature_{}'.format(n) for n in range(X.shape[1])]
        df['outcome'] = y
        return df

class SyntheticDSShift(SyntheticScenario):
    data_source_outcome_col = field(str, initial='target')
    weights = pvector_field(float, initial=[0.95, 0.04, 0.01])

    def load_data(self):
        raise NotImplemented

    @property
    def dataset_shifted(self) -> bool:
        return True

    def train_test(self, legacy_size=0.7, legacy_drop_frac=0.5, legacy_missing_label=0.2):
        X, y = make_classification(
            n_samples=100000,
            n_classes=len(self.weights),
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_repeated=self.n_repeated,
            n_clusters_per_class=self.n_clusters_per_class,
            weights=list(self.weights),
        )

        df = pd.DataFrame(X)
        df.columns=['feature_{}'.format(n) for n in range(X.shape[1])]
        df['raw_target'] = y
        del y
        # Dataset has 2 primary types, plus a third one which is distributed
        # differently but gets lumped into the second
        df[self.data_source_outcome_col] = (df['raw_target'] > 0)

        # We will adjust the data so that the new cluster only shows up in
        # new data (non-legacy)
        df_new = df[df['raw_target'] == 2].copy()
        df_old = df[df['raw_target'] != 2].copy()
        del df

        df_old['legacy'] = bernoulli(legacy_size).rvs(len(df_old))
        # New data is by definition not legacy
        df_new['legacy'] = 0

        X = pd.concat([df_old, df_new])

        y = X[self.data_source_outcome_col]
        X.drop(columns=[self.data_source_outcome_col, 'raw_target'], inplace=True)

        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75)


        train_y -= train_X['legacy'] * train_y * bernoulli(legacy_missing_label).rvs(len(train_y))

        for c in train_X.columns:
            if (bernoulli(legacy_drop_frac).rvs() == 1) and (c != 'legacy'):
                train_X.loc[train_X['legacy'] == 1, c] = None

        return train_X, test_X, train_y, test_y


class CoverTypeBinarized(ModelScenario):
    data_source_outcome_col = field(str, initial='target')

    @property
    def data_source_description(self):
        return 'sklearn.datasets.fetch_covtype, modified'

    @property
    def dataset_shifted(self) -> bool:
        return True

    def train_test(self, legacy_size=0.7, legacy_drop_frac=0.5, legacy_missing_label=0.2):
        x = fetch_covtype(as_frame=True)
        df = x['data']
        df['raw_target'] = x['target']
        # Dataset has 2 primary cover types, plus 7 more less common ones
        # so lets put some nice imbalance in our set
        df['target'] = (df['raw_target'] >= 3)

        # We will adjust the data so that the 7'th cover type only shows up
        # in non-legacy data.
        df_new = df[df['raw_target'] == 7].copy()
        df_old = df[df['raw_target'] != 7].copy()
        del df

        df_old['legacy'] = bernoulli(legacy_size).rvs(len(df_old))
        df_new['legacy'] = 0

        X = pd.concat([df_old, df_new])

        y = X[self.data_source_outcome_col]
        X = X.drop(columns=[self.data_source_outcome_col, 'raw_target'])

        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75)

        train_y -= train_X['legacy'] * train_y * bernoulli(legacy_missing_label).rvs(len(train_y))

        for c in train_X.columns:
            if (bernoulli(legacy_drop_frac).rvs() == 1) and (c != 'legacy'):
                train_X.loc[train_X['legacy'] == 1, c] = None

        return train_X, test_X, train_y, test_y


SCENARIOS = {
    'santander': ModelScenario(
        data_path='santander/santander_training_data.csv',
        data_source_name='https://www.kaggle.com/competitions/santander-customer-satisfaction',
        data_source_outcome_col='TARGET',
        data_source_id_col='ID',
    ),
    'car_insurance': _CarInsuranceModelScenario(
        data_path='car_insurance/train.csv',
        data_source_name='https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification',
        data_source_outcome_col='is_claim',
        data_source_id_col='policy_id',
    ),
    'tabular_playground': _TabularPlaygroundModelScenario(
        data_path='tabular_playground_aug_2022/train.csv',
        data_source_name='https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data?select=train.csv',
        data_source_outcome_col='failure',
        data_source_id_col='id',
    ),
    'synthetic_1': SyntheticScenario(
        data_source_name='Synthetic 1'
    ),
    'synthetic_2_dataset_shift': SyntheticDSShift(
        data_source_name='Synthetic 2 ds shift'
    ),
    'cover_type_dataset_shift': CoverTypeBinarized(
        data_source_name='cover type, ds shift',
    ),
}

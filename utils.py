import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


def preprocess(df):
    df['target'] = np.where(
        df['readmitted'] != 'NO',
        1,
        0
    )
    df = df.drop(['readmitted', 'encounter_id', 'patient_nbr', 'weight', 'race', 'gender'], axis=1)

    df['admission_type_id'] = df['admission_type_id'].astype(str)
    df['discharge_disposition_id'] = df['discharge_disposition_id'].astype(str)
    df['admission_source_id'] = df['admission_source_id'].astype(str)

    under_40 = ['[0-10)', '[10-20)', '[20-30)', '[30-40)']
    under_60 = ['[40-50)', '[50-60)']
    df['age_group'] = np.where(
        df['age'].isin(under_40),
        'under_40',
        'above_60'
    )
    df['age_group'] = np.where(
        df['age'].isin(under_60),
        'between_40_60',
        df['age_group']
    )
    df = df.drop('age', axis=1)
    return df


def fill_missing_values(df, fill_value):
    """
    Fills all missing values in a dataframe with fill_value.

    :param df: pandas dataframe
    :param fill_value: the fill value
    :returns: pandas dataframe
    """
    df = df.fillna(value=fill_value)
    df = df.replace('nan', fill_value)
    return df


class CombineCategoryLevels(BaseEstimator, TransformerMixin):
    """
    Combines category levels that individually fall below a certain percentage of the total.
    """
    def __init__(self, combine_categories='yes', sparsity_cutoff=0.01):
        self.combine_categories = combine_categories
        self.sparsity_cutoff = sparsity_cutoff
        self.mapping_dict = {}

    def fit(self, X, Y=None):
        for col in list(X):
            percentages = X[col].value_counts(normalize=True)
            combine = percentages.loc[percentages <= self.sparsity_cutoff]
            combine_levels = combine.index.tolist()
            self.mapping_dict[col] = combine_levels
        return self

    def transform(self, X, Y=None):
        if self.combine_categories == 'yes':
            for col in list(X):
                combine_cols = self.mapping_dict.get(col, [None])
                X.loc[X[col].isin(combine_cols), col] = 'sparse_combined'
            return X
        elif self.combine_categories == 'no':
            return X
        else:
            return X


class FeaturesToDict(BaseEstimator, TransformerMixin):
    """
    Converts dataframe, or numpy array, into a dictionary oriented by records. This is a necessary pre-processing step
    for DictVectorizer().
    """
    def __int__(self):
        pass

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = X.to_dict(orient='records')
        return X


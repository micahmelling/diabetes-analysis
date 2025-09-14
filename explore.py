import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from functools import partial

from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def generate_summary_statistics(df, target, save_path):
    """
    Calculates summary statistics for each target level and writes the results into FILES_PATH.

    :param df: pandas dataframe
    :param target: name of the target
    :param save_path: path in which to save the output
    """
    total_df = df.describe(include='all')
    total_df['level'] = 'total'
    results = [total_df]
    for level in (list(set(df[target].tolist()))):
        temp_df = df.loc[df[target] == level]
        temp_df = temp_df.describe(include='all')
        temp_df['level'] = level
        results.append(temp_df)
    output_df = pd.concat(results, axis=0)
    output_df.to_csv(os.path.join(save_path, 'summary_statistics.csv'), index=True)


def score_features_using_mutual_information(df, target, save_path):
    """
    Scores univariate features using mutual information. Saves the output locally.

    :param df: pandas dataframe
    :param target: name of the target
    :param save_path: path in which to save the output
    """
    print('scoring features using mutual information...')
    y = df[target]
    x = df.drop([target], axis=1)
    x_numeric = x.select_dtypes(include='number')
    x_numeric.dropna(how='all', inplace=True, axis=1)
    x_numeric = pd.DataFrame(SimpleImputer(strategy='mean', copy=False).fit_transform(x_numeric),
                             columns=list(x_numeric))
    x_categorical = x.select_dtypes(include='object')
    x_categorical = pd.get_dummies(x_categorical, dummy_na=True)

    def _fit_feature_selector(x, scorer, discrete_features):
        scorer = partial(scorer, discrete_features=discrete_features)
        feature_selector = SelectPercentile(scorer)
        _ = feature_selector.fit_transform(x, y)
        feature_scores = pd.DataFrame()
        feature_scores['score'] = feature_selector.scores_
        feature_scores['attribute'] = x.columns
        return feature_scores

    numeric_scores = _fit_feature_selector(x_numeric, mutual_info_classif, discrete_features=False)
    categorical_scores = _fit_feature_selector(x_categorical, mutual_info_classif, discrete_features=True)
    feature_scores = pd.concat([numeric_scores, categorical_scores])
    feature_scores.reset_index(inplace=True, drop=True)
    feature_scores.sort_values(by='score', ascending=False, inplace=True)
    feature_scores.to_csv(os.path.join(save_path, 'univariate_features_mutual_information.csv'), index=False)


def main():
    df = pd.read_csv('data/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv')

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

    # df = df.head(15_000)
    # df = df[['age_group', 'race', 'gender', 'admission_type_id', 'discharge_disposition_id',
    #          'admission_source_id', 'num_lab_procedures',
    #          'target']]

    score_features_using_mutual_information(df, target='target', save_path='output')

    # generate_summary_statistics(df, target='readmitted', save_path='output')
    # race	gender	age


if __name__ == "__main__":
    main()

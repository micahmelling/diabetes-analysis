import pandas as pd
from hyperopt import hp
from collections import namedtuple
from sklearn.ensemble import RandomForestClassifier
# https://endtoenddatascience.com/chapter11-machine-learning-calibration
from sklearn.calibration import CalibratedClassifierCV
import joblib
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, Trials, space_eval
from sklearn.model_selection import train_test_split
import os
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import FunctionTransformer
import sqlite3

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, VarianceThreshold
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, f1_score

from utils import preprocess, fill_missing_values, FeaturesToDict, CombineCategoryLevels


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


FOREST_PARAM_GRID = {
        'preprocessor__categorical_transformer__category_combiner__combine_categories': hp.choice(
        'preprocessor__categorical_transformer__category_combiner__combine_categories', ['yes', 'no']),
        'preprocessor__categorical_transformer__feature_selector__percentile': hp.uniformint(
        'preprocessor__categorical_transformer__feature_selector__percentile', 1, 100),
    'preprocessor__numeric_transformer__feature_selector__percentile': hp.uniformint('preprocessor__numeric_transformer__feature_selector__percentile', 1, 100),
    'model__estimator__max_depth': hp.uniformint('model__estimator__max_depth', 3, 16),
    'model__estimator__min_samples_leaf': hp.uniform('model__estimator__min_samples_leaf', 0.001, 0.01),
    'model__estimator__max_features': hp.choice('model__estimator__max_features', ['log2', 'sqrt']),
}

model_named_tuple = namedtuple('model_config', {'model_name', 'model', 'param_space', 'iterations'})
MODEL_TRAINING_LIST = [
    model_named_tuple(model_name='random_forest', model=CalibratedClassifierCV(
        estimator=RandomForestClassifier()),
        param_space=FOREST_PARAM_GRID, iterations=10),
]



def plot_calibration_curve(y_test, predictions, n_bins=10, bin_strategy='uniform'):
    """
    Produces a calibration plot and saves it locally. The raw data behind the plot is also written locally.

    :param y_test: y_test series
    :param predictions: predictions series
    :param n_bins: number of bins for the predictions
    :param bin_strategy: uniform - all bins have the same width; quantile - bins have the same number of observations
    """
    try:
        prob_true, prob_pred = calibration_curve(y_test, predictions, n_bins=n_bins, strategy=bin_strategy)
        fig, ax = plt.subplots()
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='model')
        line = mlines.Line2D([0, 1], [0, 1], color='black')
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        fig.suptitle(f' {bin_strategy.title()} Calibration Plot {n_bins} Requested Bins')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability in Each Bin')
        plt.legend()
        plt.savefig(os.path.join('{bin_strategy}_{n_bins}_calibration_plot.png'))
        plt.clf()
        calibration_df = pd.DataFrame({'prob_true': prob_true, 'prob_pred': prob_pred})
        calibration_df.to_csv(os.path.join(f'{bin_strategy}_{n_bins}_calibration_summary.csv'), index=False)
    except Exception as e:
        print(e)


def train_model(x_train, y_train, x_test, y_test, get_pipeline_function, model_uid, model, param_space, iterations,
                cv_strategy, cv_scoring):
    """
    Trains a machine learning model, optimizes the hyperparameters, and saves the serialized model into the
    MODELS_DIRECTORY.
    :param x_train: x_train dataframe
    :param y_train: y_train series
    :param get_pipeline_function: callable that takes model to produce a scikit-learn pipeline
    :param model_uid: model uid
    :param model: instantiated model
    :param param_space: the distribution of hyperparameters to search over
    :param iterations: number of trial to search for optimal hyperparameters
    :param cv_strategy: cross validation strategy
    :param cv_scoring: scoring method used for cross validation
    :param static_param_space: parameter search space valid for all models (e.g. feature engineering)
    :param db_schema_name: database schema to log metrics to
    :param db_conn: database connection
    :param log_to_db: Boolean of whether or not to log model info to a database
    :returns: scikit-learn pipeline
    """
    print(f'training {model_uid}...')
    pipeline = get_pipeline_function(model)

    cv_scores_df = pd.DataFrame()

    def _model_objective(params):
        pipeline.set_params(**params)
        score = cross_val_score(pipeline, x_train, y_train, cv=cv_strategy, scoring=cv_scoring, n_jobs=-1)

        # temp_cv_scores_df = pd.DataFrame(score)
        # temp_cv_scores_df = temp_cv_scores_df.reset_index()
        # temp_cv_scores_df['index'] = 'fold_' + temp_cv_scores_df['index'].astype(str)
        # temp_cv_scores_df = temp_cv_scores_df.T
        # temp_cv_scores_df = temp_cv_scores_df.add_prefix('fold_')
        # temp_cv_scores_df = temp_cv_scores_df.iloc[1:]
        # temp_cv_scores_df['mean'] = temp_cv_scores_df.mean(axis=1)
        # temp_cv_scores_df['std'] = temp_cv_scores_df.std(axis=1)
        # temp_params_df = pd.DataFrame(params, index=list(range(0, len(params) + 1)))
        # temp_cv_scores_df = pd.concat([temp_params_df, temp_cv_scores_df], axis=1)
        # temp_cv_scores_df = temp_cv_scores_df.dropna()
        # nonlocal cv_scores_df
        # cv_scores_df = cv_scores_df._append(temp_cv_scores_df)

        return 1 - score.mean()

    trials = Trials()
    best = fmin(_model_objective, param_space, algo=tpe.suggest, max_evals=iterations, trials=trials)
    best_params = space_eval(param_space, best)

    # cv_scores_df = cv_scores_df.sort_values(by=['mean'], ascending=False)
    # cv_scores_df = cv_scores_df.reset_index(drop=True)
    # cv_scores_df = cv_scores_df.reset_index()
    # cv_scores_df = cv_scores_df.rename(columns={'index': 'ranking'})
    # cv_scores_df.to_csv(f'cv_scores_{model_uid}.csv')

    pipeline.set_params(**best_params)
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, f'{model_uid}.pkl')

    df = pd.concat(
        [
            pd.DataFrame(pipeline.predict_proba(x_test), columns=['0_prob', '1_prob']),
            y_test.reset_index(drop=True)
        ],
        axis=1)
    df['predicted_class'] = np.where(df['1_prob'] >= 0.50, 1, 0)
    df = df[['predicted_class'] + [col for col in df.columns if col != 'predicted_class']]
    df.to_csv(f'{model_uid}_predictions_vs_actuals.csv', index=False)

    score = log_loss(df['target'], df['1_prob'])
    pd.DataFrame({'log_loss': [score]}).to_csv(f'{model_uid}_log_loss.csv', index=False)
    score = roc_auc_score(df['target'], df['1_prob'])
    pd.DataFrame({'roc_auc': [score]}).to_csv(f'{model_uid}_roc_auc.csv', index=False)
    f1 = f1_score(df['target'], df['predicted_class'])
    pd.DataFrame({'f1': [f1]}).to_csv(f'{model_uid}_f1.csv', index=False)

    plot_calibration_curve(y_test, predictions=df['1_prob'])

    return pipeline





def get_pipeline(model):
    """
    Generates a scikit-learn modeling pipeline with model as the final step.

    :param model: instantiated model
    :returns: scikit-learn pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer(sparse=False)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selector', SelectPercentile(f_classif)),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', FunctionTransformer(fill_missing_values, validate=False,
                                        kw_args={'fill_value': 'missing'})),
        ('category_combiner', CombineCategoryLevels()),
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer(sparse=False)),
        ('feature_selector', SelectPercentile(chi2)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_transformer', numeric_transformer, selector(dtype_include='number')),
            ('categorical_transformer', categorical_transformer, selector(dtype_exclude='number'))
        ],
        remainder='passthrough',
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('variance_thresholder', VarianceThreshold()),
        ('model', model)
    ])

    return pipeline


def read_data():
    return pd.read_csv('data/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv')


def main():
    df = read_data()
    df = preprocess(df)

    # conn = sqlite3.connect('diabetes.db')
    # df.to_sql('diabetes', conn, if_exists='replace', index=False)
    # import sys
    # sys.exit()

    df = df.head(20_000)
    df = df[['target', 'age_group', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
             'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications']]

    y_df = df['target']
    x_df = df.drop('target', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.25, random_state=19)

    for model in MODEL_TRAINING_LIST:
        train_model(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            get_pipeline_function=get_pipeline,
            model_uid=model.model_name,
            model=model.model,
            param_space=model.param_space,
            iterations=model.iterations,
            cv_strategy=5,
            cv_scoring='neg_log_loss'
        )


if __name__ == "__main__":
    main()

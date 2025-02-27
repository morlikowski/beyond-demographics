import math
import json 
import logging

from tqdm import tqdm
import fire
import pandas as pd
import numpy as np
import scipy
from transformers import EvalPrediction
from pandas.api.types import is_string_dtype
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import json

import fire

from multi_annotator.utils import (
    task_to_scale_max,
    normalize_int_label,
    to_simplified_output_name,
    map_predictions_to_discrete_scale
)

DEFAULT_LR = 3e-4

def gather_attribute_settings(attributes_dict, key=None):
    attribute_settings = []
    for attribute, settings in attributes_dict.items():
        for setting, value in settings.items():
            if not key or setting == key:
                attribute_settings.append(f'{attribute}-{setting}-{value}')
    return attribute_settings

def generate_metrics_config(
        config_path : str
    ):

    with open(config_path) as file:
        config = json.load(file)

    defaults = config['defaults']
    settings = config['settings']
    metrics_config = {}
    metrics_config['is_classification'] = defaults['do_train_as_classification']
    metrics_config['variables'] = ['tasks', 'seed', 'attributes', 'init', 'num_tokens', 'num_epochs', 'use_description', 'lr']
    metrics_config['outcomes'] = []

    for setting in settings:
        s = defaults.copy()
        s.update(setting)
        do_binarize = s['binarize_labels'] 
        output_model_name = s['output_model_name']
        seed = s['seed']
        do_prompt = s['do_prompt'] if 'do_prompt' in s else False
        tasks = s['tasks']
        if 'eval_tasks' in s:
            tasks = s['eval_tasks']
        learning_rate = s['learning_rate'] if 'learning_rate' in s else DEFAULT_LR
        eval_split = s['eval_split']  if 'eval_split' in s else 'test'
        output_name = to_simplified_output_name(
            seed, 
            output_model_name, 
            tasks, 
            s['annotator_attributes'], 
            s['split_based_on'], 
            learning_rate, 
            do_prompt,
            eval_split
        )
        attributes = [attribute for attribute in s['annotator_attributes'].keys()]
        attributes = ', '.join(attributes) if attributes else 'none'
        attribute_inits = gather_attribute_settings(s['annotator_attributes'], key='init')
        attribute_inits = ', '.join(attribute_inits)  if attribute_inits else 'none'
        attribute_num_tokens = gather_attribute_settings(s['annotator_attributes'], key='num_tokens')
        attribute_num_tokens = ', '.join(attribute_num_tokens)  if attribute_num_tokens else 'none'
        attribute_use_descs = gather_attribute_settings(s['annotator_attributes'], key='use_description')
        attribute_use_descs = ', '.join(attribute_use_descs) if attribute_use_descs else 'none'
        tasks = ', '.join(s['tasks'])
        num_epochs = s['num_epochs'] if 'num_epochs' in settings else 4 # we just know the hard-coded default
        outcome = {
                "predictions_path": f'data/out/{output_model_name}/{output_name}.csv',
                "variable_values": [tasks, seed, attributes, attribute_inits, attribute_num_tokens, num_epochs, attribute_use_descs, learning_rate]
            }
        metrics_config['outcomes'].append(outcome)
    
    return metrics_config


def binarize(rating):
    if rating > 3.0:
        return 1
    else:
        return 0

def clean_gen(label):
    if not type(label) == str and math.isnan(label):
        return ''
    if 'benign' in label:
        return 'benign'
    elif 'offensive' in label:
        return 'offensive'
    else:
        return label
    
def map_cls(label):
    return int(label.split('_')[1])

def label2id(label):
    if label == 'BENIGN':
        return 0
    elif label == 'OFFENSIVE':
        return 1
    else:
        raise ValueError 

def map_plabel(label):
    if 'benign' == label:
        return 0
    elif 'offensive' == label:
        return 1

def compute_regression_metrics(pred: EvalPrediction) -> dict:
    predictions: np.ndarray = pred.predictions
    label_ids: np.ndarray = pred.label_ids

    metrics = {}
    metrics['mae'] = mean_absolute_error(label_ids, predictions)
    metrics['mse'] = mean_squared_error(label_ids, predictions)
    metrics['r2'] = r2_score(label_ids, predictions)
    return metrics

def compute_metrics(df, scale_max, label_column='label', do_classification_metrics=False, ):
    result = {}
     # TODO configure binarize
    #df['binary'] = df[label_column].apply(binarize)

    if df[label_column].min() > 0:
        raise ValueError('Labels not starting from 0 used in comparison')
    if df['prediction'].min() > 0:
        raise ValueError('Predictions not starting from 0 used in comparison')
    
    if do_classification_metrics or df['prediction'].dtype != 'float64':
        if df['prediction'].dtype == 'float64':
            labels = df[label_column].apply(str)
            predictions = df['prediction'].apply(str)
        else:
            labels = df[label_column]
            predictions = df['prediction']

        report = classification_report(
            labels,
            predictions,
            zero_division=0,
            output_dict=True
        )
        result['f1'] = report['macro avg']['f1-score']
        result['accuracy'] = report['accuracy']

        most_frequent_label = df['label'].mode().item()
        if type(most_frequent_label) == float:
            most_frequent_label = str(most_frequent_label)
        baseline_report = classification_report(
            labels,
            len(predictions) * [most_frequent_label],
            zero_division=0,
            output_dict=True
        )
        result['baseline_f1'] = baseline_report['macro avg']['f1-score']

    if df['prediction'].dtype == 'int64':
        labels_cont = df[label_column].apply(normalize_int_label, scale_max=scale_max)
        predictions_cont = df['prediction'].apply(normalize_int_label, scale_max=scale_max)
    else:
        labels_cont = df[label_column]
        predictions_cont = df['prediction']

    result['r'] = labels_cont.corr(predictions_cont)
    result['rho'] = labels_cont.corr(predictions_cont, method='spearman')
    result['mae'] = mean_absolute_error(labels_cont, predictions_cont)
    # cheating baseline because mean of test label, not train label
    result['baseline_mae'] = mean_absolute_error(labels_cont, [labels_cont.mean()] * len(df.index))
    result['baseline_mse'] = mean_squared_error(labels_cont, [labels_cont.mean()] * len(df.index))
    result['baseline_r2'] = r2_score(labels_cont, [labels_cont.mean()] * len(df.index))
    result['mse'] = mean_squared_error(labels_cont, predictions_cont)
    result['r2'] = r2_score(labels_cont, predictions_cont)

    for attribute in ['race', 'education', 'age', 'gender']:
        for name, group in df.groupby(attribute):
            name_str = name.lower().replace(' ', '_').replace('/', '_')
            if do_classification_metrics or group['prediction'].dtype != 'float64':
                if group['prediction'].dtype == 'float64':
                    labels = group[label_column].apply(str)
                    predictions = group['prediction'].apply(str)
                else:
                    labels = group[label_column]
                    predictions = group['prediction']

                group_report = classification_report(
                    labels,
                    predictions,
                    zero_division=0,
                    output_dict=True
                )
                result[f'{name_str}_f1'] = group_report['macro avg']['f1-score']

            result[f'{name_str}_support'] = len(group.index)

            if df['prediction'].dtype == 'int64':
                group_labels_cont = group[label_column] / scale_max
                group_predictions_cont = group['prediction'] / scale_max
            else:
                group_labels_cont = group[label_column]
                group_predictions_cont = group['prediction']

            if group[label_column].nunique() > 1 and group_predictions_cont.nunique() > 1:
                result[f'{name_str}_r'] = group_labels_cont.corr(group_predictions_cont)
                result[f'{name_str}_rho'] = group_labels_cont.corr(group_predictions_cont, method='spearman')
            else:
                result[f'{name_str}_r'] = float('nan')
                result[f'{name_str}_rho'] = float('nan')

            result[f'{name_str}_mse'] = mean_squared_error(group_labels_cont, group_predictions_cont)
            result[f'{name_str}_r2'] = r2_score(group_labels_cont, group_predictions_cont)
            result[f'{name_str}_mae'] = mean_absolute_error(group_labels_cont,group_predictions_cont)
            # cheating baseline because mean of test label, not train label
            result[f'{name_str}_baseline_mae'] = mean_absolute_error(group_labels_cont, [group_labels_cont.mean()] * len(group.index))
            result[f'{name_str}_baseline_mse'] = mean_squared_error(group_labels_cont, [group_labels_cont.mean()] * len(group.index))
            result[f'{name_str}_baseline_r2'] = r2_score(group_labels_cont, [group_labels_cont.mean()] * len(group.index))
    return result


def calculate_metric_cis_normal(df_results, metric = 'f1'):
    """Calculate confidence intervals based on the assumption that sample means are normaly distributed.
    Here, each sample is a run with a specific random seed.

    Derived from:
    https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html#method-4-confidence-intervals-from-retraining-models-with-different-random-seeds

    df_results - Dataframe that contains results from runs with different random seeds. Needs to have columns seed and value of metric keyword argument
    metric - String naming the column for which to calculate confidence intervals
    """
    # 
    confidence = 0.95  # Change to your desired confidence level
    n = len(df_results.index)
    n_unique_seeds = df_results['seed'].nunique()

    if n != n_unique_seeds:
        raise ValueError('Each result needs to come from a unique random seed but n != n_unique_seeds')
    
    test_scores = df_results[metric]
    test_mean = test_scores.mean()

    t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=n_unique_seeds - 1)

    sd = np.std(test_scores, ddof=1)
    se = sd / np.sqrt(n_unique_seeds)

    ci_length = t_value * se

    ci_lower = test_mean - ci_length
    ci_upper = test_mean + ci_length

    group_bootstrap_cis = {}
    group_bootstrap_cis[f'mean_{metric}_ci_lower'] = ci_lower
    group_bootstrap_cis[f'mean_{metric}_ci_upper'] = ci_upper
    return group_bootstrap_cis
    


def calculate_metric_cis_bootstrap(df_results, num_bootstrap_samples=10_000):
    """Boostrapping confidence intervals of a score mean over several random seeds. 
    
    df_results - Dataframe that contains results from runs with different random seeds. Needs to have columns seed, f1, mae, mse, r2
    num_bootstrap_samples - Number of iterations of sampling
    """
    n = len(df_results.index)
    n_unique_seeds = df_results['seed'].nunique()

    if n != n_unique_seeds:
        raise ValueError('Each result needs to come from a unique random seed but n != n_unique_seeds')

    bootstrap_samples = []
    for _ in tqdm(range(num_bootstrap_samples)):
        sample_metrics = {}
        sample = df_results.sample(n = n, replace=True)

        metric_cols = [c for c in sample.columns if any(metric in c for metric in ['mae', 'mse', 'r2', 'f1'])]
        for col in metric_cols:
            sample_metrics[f'mean_{col}'] = sample[col].mean()
        bootstrap_samples.append(sample_metrics)
    
    group_bootstrap_cis = {}
    df_bootstrap_samples = pd.DataFrame.from_records(bootstrap_samples)
    for c in df_bootstrap_samples.columns:
        ci_lower = df_bootstrap_samples[c].quantile(q=0.025)
        ci_upper = df_bootstrap_samples[c].quantile(q=0.975)
        group_bootstrap_cis[f'{c}_ci_lower'] = ci_lower
        group_bootstrap_cis[f'{c}_ci_upper'] = ci_upper

    return group_bootstrap_cis

#TODO gen and labelp
def save_metrics_cls(
        experiment_config_path,
        out_path,
        do_calculate_cis=True,
        is_discrete=False
    ):

    config = generate_metrics_config(experiment_config_path)

    variables = config['variables']
    outcomes = config['outcomes']

    results = []
    all_dfs = []
    for outcome in outcomes:

        result = {var: value for var, value in zip(variables, outcome['variable_values'])}
        result['predictions_path'] = outcome['predictions_path']
        try:
            df = pd.read_csv(outcome['predictions_path'])
            task_name = df['task'].unique()[0]
            scale_max = task_to_scale_max(task_name)
            # NOTE older prompting results (before November 27) have an error in their label mapping (generation to numberic) 
            # we correct for this here
            if 'generation' in df.columns:
                from multi_annotator.utils import map_letter_labels_to_numeric
                df['prediction'] = df['generation'].apply(map_letter_labels_to_numeric, scale_1_to=scale_max)
            if not config['is_classification'] and is_discrete:
                df['prediction'] = df['prediction'].apply(map_predictions_to_discrete_scale, scale_max=scale_max)
            all_dfs.append(df)
        except FileNotFoundError:
            logging.error(f'No predictions found for {outcome["predictions_path"]}')
            result['f1'] = float('nan')
            results.append(result)
            continue
        result = {var: value for var, value in zip(variables, outcome['variable_values'])}
        result.update(
            compute_metrics(
                df,
                scale_max=scale_max,
                do_classification_metrics=config['is_classification'] or is_discrete
            )
        )
        results.append(result)
   
    df_results = pd.DataFrame.from_records(results)
    if do_calculate_cis:
        group_bootstrap_cis = calculate_metric_cis_bootstrap(df_results)
        df_results = df_results.assign(**group_bootstrap_cis)

    if out_path == 'clipboard':
        df_results.to_clipboard()
        logging.info(f'Metrics for {experiment_config_path} copied to clipboard')
    else:
        df_results.to_csv(out_path, index=False, )
        logging.info(f'Metrics written to {out_path}')

if __name__ == '__main__':
    fire.Fire(save_metrics_cls)
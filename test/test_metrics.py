import pandas as pd

from multi_annotator.metrics import (
    calculate_metric_cis_bootstrap, 
    calculate_metric_cis_normal, 
    compute_metrics
)
from multi_annotator.utils import task_to_scale_max 
from multi_annotator.utils import (
    map_letter_labels_to_numeric,
    map_predictions_to_discrete_scale
)

def test_map_predictions_to_3_point_scale():
    test_mapping = [
        (0.20703125, 0.0),
        (0.4501235, 0.5),
        (0.720135, 0.5),
        (0.822131, 1.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (0.0, 0.0),
        (-2343.312, 0.0),
        (2343.312, 1.0)
    ]
    for prediction, expected_discrete in test_mapping:
        # scale from 1 to 3 so normalized 0.0, 0.5, 1.0
        discrete = map_predictions_to_discrete_scale(prediction, scale_max=3)
        assert discrete == expected_discrete

def test_map_predictions_to_5_point_scale():
    test_mapping = [
        (0.20703125, 0.25),
        (0.4501235, 0.5),
        (0.720135, 0.75),
        (0.822131, 0.75),
        (0.89212, 1.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (0.0, 0.0),
        (-2343.312, 0.0),
        (2343.312, 1.0)
    ]
    for prediction, expected_discrete in test_mapping:
        # scale from 1 to 5 so normalized 0.0, 0.25, 0.5, 0.75, 1.0
        discrete = map_predictions_to_discrete_scale(prediction, scale_max=5)
        assert discrete == expected_discrete

def test_metrics_from_discrete_float_predictions_as_classification():
    df = pd.read_csv('data/out/llama3/536804-llama3-intimacy-notokens-instance-0.0003.csv')
    scale_max = task_to_scale_max('intimacy')
    df['prediction'] = df['prediction'].apply(map_predictions_to_discrete_scale, scale_max=scale_max)
    result = compute_metrics(df, do_classification_metrics=True, scale_max=scale_max)
    assert result['f1'] > 0.0 

def test_cis_from_discrete_float_predictions_as_classification():
    df = pd.read_csv('data/out/llama3/536804-llama3-intimacy-notokens-instance-0.0003.csv')
    scale_max = task_to_scale_max('intimacy')
    df['prediction'] = df['prediction'].apply(map_predictions_to_discrete_scale, scale_max=scale_max)
    result = compute_metrics(df, do_classification_metrics=True, scale_max=scale_max)
    df_results = pd.DataFrame.from_records([result])
    df_results['seed'] = 536804
    group_bootstrap_cis = calculate_metric_cis_bootstrap(df_results, num_bootstrap_samples=5)
    assert group_bootstrap_cis['mean_f1_ci_lower'] > 0.0
    assert group_bootstrap_cis['mean_f1_ci_upper'] > 0.0

def test_mse_from_classification_in_normalized_range():
    df = pd.read_csv('data/out/llama3_classification/536804-llama3_classification-intimacy-notokens-instance-0.0003.csv')
    scale_max = task_to_scale_max('intimacy')
    result = compute_metrics(df, do_classification_metrics=True, scale_max=scale_max)
    assert result['mse'] < 1.0

def test_compute_classification_metrics_from_prompted_results():
    df = pd.read_csv('data/out/llama3_instruct_long/536804-llama3_instruct_long-intimacy-textual-instance-0.0003-prompt.csv')
    scale_max = task_to_scale_max('intimacy')
    df['prediction'] = df['generation'].apply(map_letter_labels_to_numeric, scale_1_to=scale_max)
    result = compute_metrics(df, do_classification_metrics=True, scale_max=scale_max)
    assert result

def test_cis_normal():
    df_results = pd.DataFrame(data={
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'f1': [.50, .51, .45, .51, .50, .52, .49, .48, .50, .51]
    })
    group_bootstrap_cis = calculate_metric_cis_normal(df_results, metric='f1')
    assert group_bootstrap_cis['mean_f1_ci_lower'] > 0.0
    assert group_bootstrap_cis['mean_f1_ci_upper'] > 0.0

def test_cis_boostrap():
    df_results = pd.DataFrame(data={
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'f1': [.50, .51, .45, .51, .50, .52, .49, .48, .50, .51]
    })
    group_bootstrap_cis = calculate_metric_cis_bootstrap(df_results, num_bootstrap_samples=5)
    assert group_bootstrap_cis['mean_f1_ci_lower'] > 0.0
    assert group_bootstrap_cis['mean_f1_ci_upper'] > 0.0


def test_cis_normal():
    df_results = pd.DataFrame(data={
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'f1': [.50, .51, .45, .51, .50, .52, .49, .48, .50, .51]
    })
    group_normal_cis = calculate_metric_cis_normal(df_results, metric='f1')
    assert group_normal_cis['mean_f1_ci_lower'] > 0.0
    assert group_normal_cis['mean_f1_ci_upper'] > 0.0

def test_cis_are_similar_across_methods():
    df_results = pd.DataFrame(data={
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'f1': [.50, .51, .45, .51, .50, .52, .49, .48, .50, .51]
    })
    group_bootstrap_cis = calculate_metric_cis_bootstrap(df_results, num_bootstrap_samples=1000)
    group_normal_cis = calculate_metric_cis_normal(df_results, metric='f1')

    lower_diff = group_normal_cis['mean_f1_ci_lower'] - group_bootstrap_cis['mean_f1_ci_lower']
    upper_diff = group_normal_cis['mean_f1_ci_upper'] - group_bootstrap_cis['mean_f1_ci_upper']

    assert abs(lower_diff) < 0.005
    assert abs(upper_diff) < 0.005
 

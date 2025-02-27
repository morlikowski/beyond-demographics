import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def select_from_config_rq1(row):
    config = row['config'].lower()
    if 'non-chat' in config:
        return False
    if 'special' in config:
        return False
    return True

def map_config_name_rq1(row):
    config = row['config'].lower() 
    if 'instruct' in config:
        method = 'Prompted'
    elif 'classification' in config:
        method = 'Classification'
    elif 'discretized regression' in config:
        method = 'Discretized Regression'
    else:
        raise ValueError(config)
    
    if 'textual' in config:
        if 'long' in config:    
            attributes = 'Attributes Long'
        else:
            attributes = 'Attributes'
    elif 'content' in config:
        attributes = 'Content Only'
    else:
        raise ValueError(config)
    
    if '70b' in config:
        row['config'] = f'{method}, 70B ({attributes})'
    else:
        row['config'] = f'{method} ({attributes})'
    
    return row


def get_plot_df_rq1_(df, dataset):
    df_plot = df[ 
        (df['tasks'] == dataset) & 
        (df.apply(select_from_config_rq1, axis=1))
    ].reset_index(drop=True).apply(map_config_name_rq1, axis=1)

    df_plot = df_plot.rename(columns={'split': 'Split'})
    df_plot['Split'] = df_plot['Split'].apply(lambda val: val.title())
    return df_plot


def select_lowest_if_multiple(val):
    if type(val) == np.ndarray:
        return val[0]
    else:
        return val
    
def mean_if_multiple(val):
    if type(val) == np.ndarray:
        return val.mean()
    else:
        return val
    
def get_majority_label(row, majority_labels):
    instance = row['instance_id']
    return majority_labels[instance]

def compute_baselines(dataset, metric='mse', split='instance', debug=False):
    seed = 536804
    # workaround for temporarly missing results
    if dataset == 'dices-990':
        seed = 873469724
    test_df = pd.read_csv(f'../../data/out/llama3/{seed}-llama3-{dataset}-notokens-{split}-0.0003.csv')
    if debug:
        print(test_df.groupby('instance_id').size().describe())
    if split == 'instance':
        majority_labels = test_df.groupby('instance_id')['label'].agg(pd.Series.mode).apply(mean_if_multiple)
        test_df['majority'] = test_df['instance_id'].apply(lambda _id: majority_labels[_id])

    mean_labels = test_df.groupby('instance_id')['label'].mean()
    test_df['mean'] = test_df['instance_id'].apply(lambda _id: mean_labels[_id])
    most_frequent_label = test_df['label'].mode().item()
    if debug:
        print(most_frequent_label)
    if metric == 'mse':
        if split == 'instance':
            majorities = mean_squared_error(test_df['label'], test_df['majority'])
        mean = mean_squared_error(test_df['label'], test_df['mean'])
        most_frequent = mean_squared_error(test_df['label'], [most_frequent_label] * test_df.shape[0])
    elif metric == 'mae':
        if split == 'instance':
            majorities = mean_absolute_error(test_df['label'], test_df['majority'])
        mean = mean_absolute_error(test_df['label'], test_df['mean'])
        most_frequent = mean_absolute_error(test_df['label'], [most_frequent_label] * test_df.shape[0])
    elif metric == 'r2':
        if split == 'instance':
            majorities = r2_score(test_df['label'], test_df['majority'])
        mean = r2_score(test_df['label'], test_df['mean'])
        most_frequent = r2_score(test_df['label'], [most_frequent_label] * test_df.shape[0])
    elif metric == 'r':
        if split == 'instance':
            majorities = test_df['label'].corr(test_df['majority'])
        mean = test_df['label'].corr(test_df['mean'])
        most_frequent = test_df['label'].corr(pd.Series([most_frequent_label] * test_df.shape[0]))
    elif metric == 'rho':
        if split == 'instance':
            majorities = test_df['label'].corr(test_df['majority'], method='spearman')
        mean = test_df['label'].corr(test_df['mean'],  method='spearman')
        most_frequent = test_df['label'].corr(pd.Series([most_frequent_label] * test_df.shape[0]), method='spearman')
    else:
        raise NotImplementedError()

    if split == 'instance':
        return majorities, most_frequent, mean
    else:
        return most_frequent, mean

def select_from_config_rq2(row):
    config = row['config'].lower()
    if 'non-chat' in config:
        return False
    if 'instruct' in config and 'instance' in config:
        return True
    return False

def map_config_name_rq2(row):
    config = row['config'].lower() 
    
    if 'textual' in config:
        if 'long' in config:    
            attributes = '+Attributes'
        else:
            attributes = 'Attributes List'
    elif 'content' in config:
        attributes = 'Content Only'
    else:
        raise ValueError(config)
    
    if '70b' in config:
        row['config'] = f'{attributes} (70B)'
    else:
        row['config'] = f'{attributes}'
    
    return row


def map_task_names(task):
    if 'diaz' == task:
        return 'Sentiment'
    elif 'dices' in task:
        return 'Safety'
    else:
        return task.title()

def get_plot_df_rq2(df):
    df_plot = df[ 
        df.apply(select_from_config_rq2, axis=1)
    ].reset_index(drop=True).apply(map_config_name_rq2, axis=1)
    df_plot['tasks'] = df_plot['tasks'].apply(map_task_names)
    return df_plot

def select_from_config_rq3(row, only_llama=True, include_best_zero_shot=True):
    config = row['config'].lower()
    if 'non-chat' in config:
        return False
    
    if only_llama and 'mistral' in config:
        return False
    
    if only_llama and 'qwen' in config:
        return False
    
    ## select best zero-shot
    if '70b' in config:
        return False
    if include_best_zero_shot and 'instruct' in config and 'instance' in config:
        if 'politeness' in config:
            if 'textual' in config and 'long' in config:
                return True
        else:
            if 'content only' in config:
                return True
    
    # select classification FTed models
    if 'classification' in config and 'instance' in config and not 'special' in config:
        return True
    return False

def map_config_name_rq3(row, add_model=False):
    config = row['config'].lower()
    if 'instruct' in config:
        row['config'] = 'Best Zero-Shot'
        return row
    if 'id+attributes' in config:
        attributes = '+ID+Attributes'
    elif 'id' in config:
        attributes = '+ID'
    elif 'textual' in config:
        attributes = '+Attributes'
    elif 'content only' in config:
        attributes = 'Content Only'
    else:
        raise ValueError(config)
        

    if add_model:
        model = config.split(',')[0]
        row['config'] = f'{attributes}, {model} (FTed)'
    else:
        row['config'] = f'{attributes} (FTed)'

    return row

def get_plot_df_rq3(df):
    df_plot = df[ 
        df.apply(select_from_config_rq3, axis=1)
    ].reset_index(drop=True).apply(map_config_name_rq3, axis=1)
    df_plot['tasks'] = df_plot['tasks'].apply(map_task_names)
    return df_plot

def get_plot_df_rq3_1(df):
    df_plot = df[ 
        df.apply(select_from_config_rq3, only_llama=False, include_best_zero_shot=False, axis=1)
    ].reset_index(drop=True).apply(map_config_name_rq3, add_model=True, axis=1)
    df_plot['tasks'] = df_plot['tasks'].apply(map_task_names)
    # Mistral is only tested with 10 seeds, select same seeds for all models (Llama 3 has 30 seeds in total)
    is_experiment_seed = df_plot['seed'].isin([1124617826, 1435485536, 1506676066, 2454110510, 2591124800, 2969282657, 3208936010, 536804, 621609371, 701702170])
    is_selected_tasks = df_plot['tasks'].isin(['Intimacy', 'Offensiveness'])
    df_plot = df_plot[is_experiment_seed & is_selected_tasks].copy()
    df_plot['model'] = df_plot['config'].apply(lambda config: config.split(',')[1].strip().split(' ')[0].title())
    df_plot['model'] = df_plot['model'] + ' - ' + df_plot['tasks']
    df_plot['config'] = df_plot['config'].apply(lambda config: config.split(',')[0].strip())
    return df_plot

def select_from_config_rq4(row):
    config = row['config'].lower()
    if 'non-chat' in config:
        return False
    if '70b' in config:
        return False
    if 'instruct' in config and 'user' in config:
        if 'politeness' in config or 'dices-350' in config:
            if 'textual' in config and 'long' in config:
                return True
        else:
            if 'content only' in config:
                return True
    if 'classification' in config and 'user' in config and not 'special' in config:
        return True
    return False

def map_config_name_rq4(row):
    config = row['config'].lower() 
    if 'instruct' in config:
        row['config'] = 'Best Zero-Shot'
        return row
    
    if 'id+attributes' in config:
        attributes = '+ID+Attributes'
    elif 'id' in config:
        attributes = '+ID'
    elif 'textual' in config:
        attributes = '+Attributes'
    elif 'content only' in config:
        attributes = 'Content Only'
    else:
        raise ValueError(config)
        
    row['config'] = f'{attributes} (FTed)'
    
    return row

def get_plot_df_rq4(df):
    df_plot = df[ 
        df.apply(select_from_config_rq4, axis=1)
    ].reset_index(drop=True).apply(map_config_name_rq4, axis=1)
    df_plot['tasks'] = df_plot['tasks'].apply(map_task_names)
    return df_plot

def do_paper_plot_points(df, rq, is_paper=False):
    ylim = (0.19,0.53)
    if rq == 2:
        x = 'tasks'
        ylim = (0.19,0.35)
        plot_df = get_plot_df_rq2(df)
        palette = sns.color_palette("Paired")
        palette = [
            palette[0],
            palette[1],
            palette[4],
            palette[5],
            (0.7,0.7,0.7), # grey
        ]
        hue_order = ['Content Only (70B)', 'Content Only', '+Attributes (70B)', '+Attributes', 'Attributes List']
    elif rq == 3:
        x = 'tasks'
        plot_df = get_plot_df_rq3(df)
        palette = sns.color_palette("Paired")
        palette = [
            palette[7],
            palette[1], 
            palette[6],
            palette[5],
            (0.7,0.7,0.7), # grey
        ]
        hue_order = ['+ID (FTed)',  'Content Only (FTed)', '+ID+Attributes (FTed)', '+Attributes (FTed)', 'Best Zero-Shot']
    elif rq == 3.1:
        x = 'model'
        ylim = (0.15,0.46)
        plot_df = get_plot_df_rq3_1(df)
        palette = None
        # palette = sns.color_palette("Paired")
        # palette = [
        #     palette[7],
        #     palette[1], 
        #     palette[6],
        #     palette[5]
        # ]
        hue_order = None# ['+ID (FTed)',  'Content Only (FTed)', '+ID+Attributes (FTed)', '+Attributes (FTed)']
    elif rq == 4:
        x = 'tasks'
        plot_df = get_plot_df_rq4(df)
        palette = sns.color_palette("Paired")
        palette = [
            palette[7],
            palette[1], 
            palette[6],
            palette[5],
            (0.7,0.7,0.7), # grey
        ]
        hue_order = ['+ID (FTed)',  'Content Only (FTed)', '+ID+Attributes (FTed)', '+Attributes (FTed)', 'Best Zero-Shot']
    else:
        raise ValueError('Need to pass valid rq number')
    
    plot_df = plot_df.sort_values(by='tasks')

    with sns.axes_style("whitegrid", {"grid.color": ".9", "grid.linestyle": "--"}):
        ax = sns.pointplot(
            data=plot_df, y="f1", x=x, hue='config',
            hue_order=hue_order,
            errorbar=("ci", 95), 
            n_boot=10_000, 
            seed=4242424242,
            capsize=.15,
            markersize= 1.5 if is_paper else 3.5,
            err_kws={'linewidth': 1},
            palette=palette,
            linestyle="none",
            marker="o", 
            dodge=.3
        )

        ax.set_xlabel(None)
        ax.set_ylabel('F1')
        ax.set_ylim(ylim)

        if is_paper:
            ax.tick_params(axis='x', rotation=20)

        sns.move_legend(
            ax, 
            "lower center",
            bbox_to_anchor=(0.45, 1), ncol=3, title=None, frameon=False,
            labelspacing=.2,
            handlelength=1.0,
            handletextpad=.1,
            columnspacing=.7
        )

        ax.axvspan(0.5, 1.5, facecolor='#BBA1')
        ax.axvspan(2.5, 3.5, facecolor='#BBA1')

        sns.despine()

def do_paper_plot_rq1_(df, dataset, metric='mse', add_mean=False, order=None):
    df_plot = get_plot_df_rq1_(df, dataset)

    g = sns.FacetGrid(
        df_plot, 
        col='Split',
        aspect=1.5
    )

    if not order:
        order = ['Discretized Regression (Content Only)', 'Discretized Regression (Attributes)', 'Classification (Content Only)', 'Classification (Attributes)', 'Prompted (Content Only)', 'Prompted (Attributes)', 'Prompted (Attributes Long)', 'Prompted, 70B (Content Only)', 'Prompted, 70B (Attributes Long)']
    g.map(sns.boxplot, metric, 'config', order=order)
    g.set_axis_labels(metric.upper(), '')

    ax_inst, ax_anno = g.axes[0]

    most_frequent_inst = df_plot[df_plot['Split'] == 'Instance']['baseline_f1'].unique()[0]
    ax_inst.axvline(most_frequent_inst, linestyle='--', color='0.4', label='Most Fequent Class')
    ax_inst.legend()

    most_frequent_anno = df_plot[df_plot['Split'] == 'Annotator']['baseline_f1'].unique()[0]
    ax_anno.axvline(most_frequent_anno, linestyle='--', color='0.4', label='Most Fequent Class')
    plt.legend()
    
    return g

def do_paper_plot_rq2(df, metric='mse', add_mean=False, order=None, horizontal=False, palette=['tab:blue', 'tab:orange'], plot_type='box'):
    df_plot = get_plot_df_rq2(df)
    df_plot = df_plot.sort_values('tasks')

    if not order:
        order = ['Content Only', '+Attributes']


    if metric == 'f1':
        if horizontal:
            x=metric
            y='tasks'
            pad_y=-2
            pad_x=0
        else:
            x='tasks'
            y=metric
            pad_y=0
            pad_x=-2

        if plot_type == 'box':
            ax = sns.boxplot(
                df_plot,
                x=x,
                y=y,
                hue='config',
                hue_order=order,
                palette=palette,
                fliersize=2,
                linewidth=0.5
            )
        else:
            ax = sns.pointplot(
                df_plot,
                x=x,
                y=y,
                errorbar=('ci', 95),
                n_boot=5000, 
                seed=10,
                capsize=.3,
                hue='config',
                hue_order=order,
                orient='h',
                palette=['tab:grey', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:green'],
                linestyle='none'
            )


        ax.tick_params(axis='y', which='major', pad=pad_y)
        ax.tick_params(axis='x', which='major', pad=pad_x)
    else:
        ax = sns.pointplot(
            df_plot,
            x=metric,
            y='tasks',
            hue='config',
            hue_order=order,
            linestyles='none'
        )

    if horizontal:
        ax.set_xlim((0.15,0.35))
        ax.set_ylabel('')
        ax.set_xlabel(metric.upper())
    else:
        ax.set_ylim((0.15,0.35))
        ax.set_xlabel('')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=20)
    
    ax.legend().set_title(None)

    #most_frequent_inst = df_plot['baseline_f1'].unique()[0]
    #ax.axhline(most_frequent_inst, linestyle='--', color='0.4', label='Most Fequent Class')

    #plt.legend()
    
    return ax

def do_paper_plot_rq3(df, metric='mse', add_mean=False, order=None, horizontal=False, plot_type='box'):
    df_plot = get_plot_df_rq3(df)

    if not order:
        order = [
            'Best Zero-Shot',  
            'Content Only (FTed)', '+Attributes (FTed)', '+ID (FTed)', '+ID+Attributes (FTed)'
        ]

    if metric == 'f1':
        if horizontal:
            x=metric
            y='tasks'
            pad_y=-2
            pad_x=0
        else:
            x='tasks'
            y=metric
            pad_y=0
            pad_x=-2

        if plot_type == 'box':
            ax = sns.boxplot(
                df_plot,
                x=x,
                y=y,
                hue='config',
                hue_order=order,
                palette=['tab:grey', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:green'],
                fliersize=2,
                linewidth=0.5
            )
        else:
            ax = sns.pointplot(
                df_plot,
                x=x,
                y=y,
                errorbar=('ci', 95),
                n_boot=5000, 
                seed=10,
                capsize=.3,
                hue='config',
                hue_order=order,
                palette=['tab:grey', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:green'],
                linestyle='none'
            )
        ax.tick_params(axis='y', which='major', pad=pad_y)
        ax.tick_params(axis='x', which='major', pad=pad_x)
    else:
        ax = sns.pointplot(
            df_plot,
            x=metric,
            y='tasks',
            hue='config',
            hue_order=order,
            palette=['tab:grey', 'tab:blue', 'tab:orange', 'tab:purple'],
            linestyles='none'
        )

    if horizontal:
        ax.set_xlim((0.2,0.55))
        ax.set_ylabel('')
        ax.set_xlabel(metric.upper())
    else:
        ax.set_ylim((0.2,0.55))
        ax.set_xlabel('')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=20)

    #ax.legend().set_title(None)
    sns.move_legend(
        ax, 
        "lower center",
        bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False,
    )

    #most_frequent_inst = df_plot['baseline_f1'].unique()[0]
    #ax.axhline(most_frequent_inst, linestyle='--', color='0.4', label='Most Fequent Class')

    #plt.legend()
    
    return ax

def do_paper_plot_rq4(df, metric='mse', add_mean=False, order=None, horizontal=True):
    df_plot = get_plot_df_rq4(df)

    if not order:
        order = ['Best Zero-Shot',  'Content Only (FTed)', '+Attributes (FTed)', '+ID (FTed)', '+ID+Attributes (FTed)']

    if metric == 'f1':
        if horizontal:
            x=metric
            y='tasks'
            pad_y=-2
            pad_x=0
        else:
            x='tasks'
            y=metric
            pad_y=0
            pad_x=-2

        ax = sns.boxplot(
            df_plot,
            x=x,
            y=y,
            hue='config',
            hue_order=order,
            palette=['tab:grey', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:green'],
            fliersize=2,
            linewidth=0.5
        )
        ax.tick_params(axis='y', which='major', pad=pad_y)
        ax.tick_params(axis='x', which='major', pad=pad_x)
    else:
        ax = sns.pointplot(
            df_plot,
            x=metric,
            y='tasks',
            hue='config',
            hue_order=order,
            palette=['tab:grey', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:green'],
            linestyles='none'
        )


    if horizontal:
        ax.set_xlim((0.15,0.5))
        ax.set_ylabel('')
        ax.set_xlabel(metric.upper())
    else:
        ax.set_ylim((0.15,0.5))
        ax.set_xlabel('')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=20)

    #ax.legend().set_title(None)

    sns.move_legend(
        ax, 
        "lower center",
        bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False,
    )

    #most_frequent_inst = df_plot['baseline_f1'].unique()[0]
    #ax.axhline(most_frequent_inst, linestyle='--', color='0.4', label='Most Fequent Class')

    #plt.legend()
    
    return ax


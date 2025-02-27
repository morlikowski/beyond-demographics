from ast import literal_eval
import pandas as pd
import numpy as np

def settings_to_output_name(defaults, setting, seed, do_binarize):
    setting_values = []
    for k, v in setting.items():
        if not v:
            v = 'none'
        elif type(v) == list:
            v = '-'.join(v)
        elif type(v) == dict:
            [flat_dict] = pd.json_normalize(v, sep='-').to_dict(orient='records')
            v = [f'{k}-{v}' for k, v in flat_dict.items()]
            v = '-'.join(v)
        elif type(v) == int or type(v) == float:
            v = f'{k}-{v}'

        setting_values.append(v)

    values_str = '-'.join(setting_values).lower()
    default_name = setting['output_model_name'] if 'output_model_name' in setting else defaults['output_model_name']
    task_type = 'binary' if do_binarize else 'multiclass'
    output_model_name = f'{seed}-{default_name}-{task_type}-{values_str}'
    return output_model_name


def to_simplified_output_name(seed, output_model_name, tasks, attributes, split_based_on, lr, is_prompting=False, eval_split='test'):
    model = output_model_name
    if attributes:
        if 'use_description' in list(attributes.values())[0]:
            attribute_tokens = 'textual'
            if 'user_id' in attributes.keys():
                if len(attributes) == 1:
                    attribute_tokens = 'textual_id'
                else:
                    attribute_tokens = f'id+{attribute_tokens}'
        else:
            raise ValueError(f'Wrong attributes setting, got {attributes}')
    else:
        attribute_tokens = 'notokens'

    dataset = '-'.join(tasks)

    output_model_name = f'{seed}-{model}-{dataset}-{attribute_tokens}-{split_based_on}-{lr}'
    if is_prompting:
        output_model_name = f'{output_model_name}-prompt'
    if eval_split == 'val':
        output_model_name = f'{output_model_name}-eval_on_val'

    return output_model_name

def task_to_scale_max(task_name: str):
       # TODO add new tasks if using
        if 'dices' in task_name:
            scale_max = 3
        else:
            scale_max = 5
        return scale_max

def normalize_int_label(label: int, scale_max: int) -> float:
        # integer labels are 0 to 4
        # task scales are 1 to 5
        # hence, scale_max - 1
        return label / (scale_max - 1)

def map_letter_labels_to_numeric(label: str, scale_1_to = 5) -> float:
    if label.startswith('A'):
        numeric_label = 0 / (scale_1_to - 1)
    elif label.startswith('B'):
        numeric_label = 1 / (scale_1_to - 1)
    elif label.startswith('C'):
        numeric_label = 2 / (scale_1_to - 1)
    elif label.startswith('D'):
        numeric_label = 3 / (scale_1_to - 1)
    elif label.startswith('E'):
        numeric_label = 4 / (scale_1_to - 1)
    else:
        raise ValueError('Label does not start with known letter (A, B, ...)')

    if numeric_label > 1.0:
        raise ValueError('Numeric label greater than 1.0, wrong label scale or input error?')
    return numeric_label

def map_predictions_to_discrete_scale(prediction: float, scale_max: int) -> float:
    if scale_max == 3:
        scale_values = [0.0, 0.5, 1.0]
    elif  scale_max == 5:
        scale_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        raise NotImplementedError()

    if prediction < 0.0:
        return 0.0
    elif prediction > 1.0:
        return 1.0
    else:
        distances = [abs(prediction - value) for value in scale_values]
        argmin = np.argmin(distances)
        return scale_values[argmin]

def _map_diaz_labels(label_dict_literal: str) -> int:
    label_dict = literal_eval(label_dict_literal)
    label_str = label_dict['sentiment']
    if label_str == 'very negative':
        return 1
    elif label_str == 'somewhat negative':
        return 2
    elif label_str == 'neutral':
        return 3
    elif label_str == 'somewhat positive':
        return 4
    elif label_str == 'very positive':
        return 5
    else:
        raise ValueError()

def _map_dices_labels(label_dict_literal: str) -> int:
    label_dict = literal_eval(label_dict_literal)
    label_str = label_dict['Q_overall']
    if label_str == 'No':
        return 1
    elif label_str == 'Unsure':
        return 2
    elif label_str == 'Yes':
        return 3
    else:
        raise ValueError()

def map_labels(row) -> int:
    task = row['task']
    if row['task'] == 'diaz':
        label = _map_diaz_labels(row['labels'])
    elif 'dices' in task:
        label = _map_dices_labels(row['labels'])
    else:
        label = int(literal_eval(row['labels'])[task])
    return label

def label_to_completion(task, label):
        if task == 'intimacy':
            if label == 0:
                return 'A) not intimate at all'
            elif label == 1:
                return 'B) barely intimate'
            elif label == 2:
                return 'C) somehat intimate'
            elif label == 3:
                return 'D) moderately intimate'
            elif label == 4:
                return 'E) very intimate'
            else:
                raise ValueError()
        elif task == 'politeness':
            if label == 0:
                return 'A) not polite at all'
            elif label == 1:
                return 'B) barely polite'
            elif label == 2:
                return 'C) somewhat polite'
            elif label == 3:
                return 'D) moderately polite'
            elif label == 4:
                return 'E) very polite'
            else:
                raise ValueError()
        elif task == 'offensiveness':
            if label == 0:
                return 'A) not offensive at all'
            elif label == 1:
                return 'B) barely offensive'
            elif label == 2:
                return 'C) somewhat offensive'
            elif label == 3:
                return 'D) moderately offensive'
            elif label == 4:
                return 'E) very offensive'
            else:
                raise ValueError()
        else:
            raise NotImplementedError()

def read_results(path, config_name):
    df = pd.read_csv(path)
    df['config'] = config_name
    return df
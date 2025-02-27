import pandas as pd
from datasets import Dataset

from multi_annotator.utils import map_labels

# in experiments, mapping is built from data. these are just defaults
AGE = {
    '___UNKNOWN_AGE___': 'Unknown',
    '___18-24___': '18-24',
    '___25-29___': '25-29',
    '___30-34___': '30-34',
    '___35-39___': '35-39',
    '___40-44___': '40-44',
    '___45-49___': '45-49',
    '___50-59___': '50-59',
    '___60-69___': '60-69',
    '___70-79___': '70-79',
    '___90-99___': '90-99',
    '___80-89___': '80-89',
    '___100+___': '100+'
}

# in experiments, mapping is built from data. these are just defaults
RACE = {
    '___UNKNOWN_RACE___': 'Unknown',
    '___OTHER_RACE___': 'Other',
    '___WHITE___': 'White',
    '___ASIAN___': 'Asian',
    '___MULTIRACIAL___': 'Multiracial',
    '___BLACK___': 'Black',
    '___HISPANIC/LATINO___': 'Hispanic/Latino',
    '___PACIFIC_ISLANDER___': 'Pacific Islander',
    '___NATIVE_AMERICAN___': 'Native American',
    '___ARAB___': 'Arab',
    '___AMERICAN_INDIA_OR_ALASKA_NATIVE___': 'American India or Alaska Native',
    '___MIDDLE_EASTERN___': 'Middle Eastern'
}

# in experiments, mapping is built from data. these are just defaults
GENDER = {
    '___UNKNOWN_GENDER___': 'Unknown',
    '___NON-BINARY___': 'Non-binary',
    '___MAN___': 'Man', 
    '___WOMAN___': 'Woman'
}

# in experiments, mapping is built from data. these are just defaults
EDUCATION = {
    '___UNKNOWN_EDUCATION___': 'Unknown', 
    '___LESS_THAN_HIGH_SCHOOL___': 'Less than high school', 
    '___HIGH_SCHOOL_OR_BELOW___': 'High school or below', 
    '___COLLEGE_DEGREE___': 'College degree', 
    '___GRADUATE_DEGREE___': 'Graduate degree', 
    '___COLLEGE_DEGREE_OR_HIGHER___': 'College degree or higher', 
    "___SOME_COLLEGE_OR_ASSOCIATE_DEGREE___": "Some college or associate's degree"
}

# in experiments, mapping is built from data. these are just defaults
VALUE2TOKEN = {
    'age': {v: [t] for t, v in AGE.items()},
    'gender': {v: [t] for t, v in GENDER.items()},
    'race': {v: [t] for t, v in RACE.items()},
    'education': {v: [t] for t, v in EDUCATION.items()}
}

# in experiments, mapping is built from data. these are just defaults
# the tokenizer is case-sensitive, so we lower case
# we assume that attribute descriptions will have been lower-cased in most instances in the training data 
VALUE2DESC = {
    'age': {v: [f"{' to '.join(v.split('-'))} years old".replace('+ years old', ' years or older').lower().replace('unknown years old', 'unknown age')] for v in AGE.values()},
    'gender': {v: [v.lower().replace('unknown', 'unknown gender')] for v in GENDER.values()},
    'race': {v: [v.lower().replace('unknown', 'unknown race').replace('other', 'other race')] for v in RACE.values()},
    'education': {v: [v.lower().replace('unknown', 'unknown education')] for v in EDUCATION.values()}
}

TASKS_WITH_TEXT = [
    'intimacy', 
    'politeness', 
    'offensiveness',
    'diaz',
    'dices-350' 
    'dices-990'
]

TASKS_WITHOUT_TEXT = [
    'yi-ding-cscw-2022'
]

TASKS = TASKS_WITH_TEXT + TASKS_WITHOUT_TEXT



def _build_value2desc(attribute_column: pd.Series):
    name = attribute_column.name
    values = attribute_column.unique()
    values = [v for v in values if type(v) == str]
    if 'unknown' not in [v.lower() for v in values]:
        values += ['unknown']
    descs = [f'{v} {name}' if v.lower() == 'unknown' else v for v in values]
    descs = [f'{v} {name}' if v.lower() == 'other' else v for v in descs]
    descs = [d.lower() for d in descs]
    if name == 'age':
        if not 'gen z' in values:
            descs = [f"{' to '.join(d.split('-'))} years old".replace('+ years old', ' years or older').lower() for d in descs]
    if name == 'user_id':
        descs = [f'{index}' for index, _ in enumerate(descs)]
    token_mapping =  {value: [desc] for value, desc in zip(values, descs)}
    return token_mapping

def _build_value2token(attribute_column: pd.Series):
    name = attribute_column.name
    values = attribute_column.unique()
    values = [v for v in values if type(v) == str]
    if 'unknown' not in [v.lower() for v in values]:
        values += ['unknown']
    tokens = [f'{name}_{v}' if v.lower() in ['unknown', 'other'] else v for v in values]
    tokens = [t.upper().replace(" ", "_").replace("'S", "") for t in tokens]
    tokens = [f'___{token}___' for token in tokens]
    token_mapping =  {value: [token] for value, token in zip(values, tokens)}
    return token_mapping

N_ALL = -1


class MergedDataset():

    def __init__(self,
                 data_path,
                 tasks = ['offensiveness'],
                 load_splits = ['train', 'val', 'test'],
                 attributes = {'age': {'init': 'first', 'num_tokens': 1}, 'gender': {'init': 'first', 'num_tokens': 1}},
                 # if token_mapping=None, build mapping from dataset
                 token_mapping = VALUE2TOKEN,
                 as_classification_tasks = True,
                 split_based_on = 'instance',
                 do_binarize = False,
                 n = N_ALL,
                 load_as_hf_dataset=True
                 ):
        df = pd.read_csv(data_path, low_memory=False)

        attribute_names = [name for name in attributes.keys()]
        attribute_num_tokens = {a: settings['num_tokens'] if 'num_tokens' in settings else 1 for a, settings in attributes.items()}
        attribute_use_desc = {a: settings['use_description'] if 'use_description' in settings else False for a, settings in attributes.items()}
        
        # filter tasks
        if tasks == 'all':
            tasks = TASKS
        elif tasks == 'with_text':
            tasks = TASKS_WITH_TEXT

        df = df[df['task'].isin(tasks)]
        # exclude bad annotators, i.e. annotators excluded from original datasets due to quality concerns
        df = df[df['bad_users'] == False]
        self.tasks = tasks

        if as_classification_tasks:
            df['label'] = df.apply(map_labels, axis=1)
            df['label'] = df['label'].astype(int)
            df['label'] = df['label'] - 1 # make five-point labels start from zero
            df = df.drop(columns=['labels'])
            if do_binarize:
                df['label'] = df['label'].apply(lambda l: 1 if l > 3 else 0)
            self.num_labels = df['label'].nunique()
        else:
            self.num_labels = 1


        # get splits
        if not split_based_on in ['user', 'instance']:
            raise ValueError()
        self.split_base_on = split_based_on
        split_col = f'{split_based_on}_split'
        if 'train' in load_splits:
            self.train = df[df[split_col] == 'train']
            if n > N_ALL:
                self.train = self.train.head(n)
            # TODO logging
            # logging.info(f"Loaded TRAIN data: {train_df.shape[0]} rows")
            if load_as_hf_dataset:
                self.train = Dataset.from_pandas(self.train)
        if 'val' in load_splits:
            self.val = df[df[split_col] == 'val']
            if n > N_ALL:
                self.val = self.val.head(n)
            # TODO logging
            # logging.info(f"Loaded TRAIN data: {val_df.shape[0]} rows")
            if load_as_hf_dataset:
                self.val = Dataset.from_pandas(self.val)
        if 'test' in load_splits:
            self.test = df[df[split_col] == 'test']
            if n > N_ALL:
                self.test = self.test.head(n)
            # TODO logging
            # logging.info(f"Loaded TRAIN data: {test_df.shape[0]} rows")
            if load_as_hf_dataset:
                self.test = Dataset.from_pandas(self.test)

        # TODO I kind of want to refactor this to a separate class or move to the tokenization/prompting code
        # set or build mapping from values to tokens
        if token_mapping:
            # use mapping
            # and filter for selected attributes
            value2token = {attr: mapping for attr, mapping in token_mapping.items() if attr in attribute_names}
        else:
            # build mapping from data for selected attributes
            value2token = {}
            for attribute in attribute_names:
                #TODO only built from train??
                #train_df = df[df[split_col] == 'train']
                values = df[attribute]
                value2token[attribute] = _build_value2token(values)
        self.value2token = value2token

        # set num tokens
        value2tokens = {}
        for attribute, mapping in self.value2token.items():
            value2tokens[attribute] = {}
            num_tokens = attribute_num_tokens[attribute]
            for value, token in mapping.items():
                value2tokens[attribute][value] = [f'{t}{i}___' for i in range(num_tokens) for t in token]
        self.value2token = value2tokens

        # replace with description tokens if set
        for attribute in attribute_names:
            if attribute_use_desc[attribute]:
                values = df[attribute]
                self.value2token[attribute] = _build_value2desc(values)

    def get_attribute_tokens(self, exclude_attributes=[]):
        tokens = []
        for attribute, mapping in self.value2token.items():
            if attribute not in exclude_attributes:
                tokens.extend(t for tokens in mapping.values() for t in tokens)
        return tokens
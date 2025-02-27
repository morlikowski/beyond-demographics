import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from multi_annotator.prompting import Prompter


def print_token_distribution(data_path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        split_special_tokens=False
    )

    prompter = Prompter([])
    template = prompter.template['prompt'].format(attributes='',text='')
    tokenized_template = tokenizer(template,
                                   padding=False,
                                   truncation=False)
    print(f'Template token number: {len(tokenized_template["input_ids"])}')

    no_tasks = ['dices-350', 'dices-990', 'yi-ding-cscw-2022']
    tasks = ['intimacy', 'politeness', 'offensiveness', 'diaz']
    df = pd.read_csv(data_path)
    df = df[df['task'].isin(tasks)]
    data = Dataset.from_pandas(df)
    #data = load_dataset('csv', data_files=data_path)

    def tokenize(example):
        return tokenizer(
            example["text"],
            padding=False,
            truncation=False
        )

    data = data.map(tokenize, batched=True)
    num_tokens = pd.Series(len(ids) for ids in data['input_ids'])
    print(num_tokens.describe())

    """
    'intimacy', 'politeness', 'offensiveness', 'diaz'
    count    111248.000000
    mean         35.934947
    std          24.646628
    min           3.000000
    25%          21.000000
    50%          31.000000
    75%          41.000000
    max         253.000000
    """


if __name__ == '__main__':
    print_token_distribution(
        'shared_data/processed/merged_data.csv',
        'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
    )
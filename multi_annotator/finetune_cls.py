import json
import logging
import os
from pathlib import Path
import sys
import time

import fire
import pandas as pd

import torch
import transformers

from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)

from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)

from multi_annotator.utils import to_simplified_output_name
from multi_annotator.data import MergedDataset
from multi_annotator.prompting import Prompter
from multi_annotator.preprocessing import (
    TokenizerWithSpecialTokens,
    AttributePromptPreprocessorForClassification
)
from multi_annotator.metrics import compute_regression_metrics

os.environ["WANDB_DISABLED"] = "true"

def decide_model_dtype(model_dtype: str, use_cpu: bool = False):
    if use_cpu:
        return torch.float32
    else:
        if model_dtype == 'bf16':
            return torch.bfloat16
        elif model_dtype == 'fp16':
            return torch.float16
        else:
            return torch.float32
        
# TODO consult on half-precision 
# https://huggingface.co/docs/transformers/model_doc/llama2
def decide_mixed_precision_training(training_dtype: str, use_cpu: bool = False):
    if use_cpu:
        return False, False
    else:
        if training_dtype == 'bf16':
            return False, True
        elif training_dtype == 'fp16':
            return True, False
        else:
            return False, False

def write_training_output(train_output, configuration_path, setting_id):
    experiment_config_path = Path(configuration_path)
    results_path = experiment_config_path.parent / f'{experiment_config_path.stem}.metrics-train.csv'
    metrics_df = pd.DataFrame.from_records([train_output.metrics])
    metrics_df['setting_index'] = setting_id
    metrics_df = metrics_df.set_index('setting_index')
    logging.info(f"Training metrics saved to {results_path}")
    do_write_header = not results_path.exists()
    metrics_df.to_csv(results_path, mode='a', header=do_write_header)

def is_greater_better(early_stopping_metric):
    """Says if a higher numerical value is better for a given metric."""
    if early_stopping_metric in ['r2']:
        return True
    elif early_stopping_metric in ['mse', 'mae']:
        return False
    else:
        raise NotImplementedError(f'unknown metric {early_stopping_metric}')

def train(
    # model parameters
    configuration_path: str,
    setting_id: int,  # this is the index of the setting in the configuration file
    model_output_dir: str,
    # training parameters
    batch_size: int = 64,
    micro_batch_size: int = 64, # NOTE makes default of 1 gradien accumulation step, prev 4
    cutoff_len: int = 300,
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # PEFT / quantization parameters
    lora_r: int = 8, # NOTE prev 4, also see https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules=None,
    # misc parameters
    log_level: str = "info",
    save_final_model: bool = True,
    use_cpu: bool = False,
    hf_token_path: str = None,
    use_optimized: bool = False,
):
    ###########################################################
    # SET UP
    ###########################################################

    # set up logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # set up batch size
    gradient_accumulation_steps = batch_size // micro_batch_size

    ###########################################################
    # LOAD TRAINING CONFIG
    ###########################################################

    # load all settings
    with open(configuration_path, "r") as f:
        config = json.load(f)

    # select default settings
    current_setting = config['defaults']     
    # overwrite with specific setting
    current_setting.update(config['settings'][setting_id])

    base_model = current_setting["base_model"]
    data_path = current_setting["data_path"]
    num_epochs = current_setting['num_epochs'] if 'num_epochs' in current_setting else 4
    n_train = current_setting["n_train"] if 'n_train' in current_setting else -1
    seed = current_setting["seed"]
    torch.manual_seed(seed)
    tasks = current_setting['tasks']

    split_based_on = 'instance'
    if 'split_based_on' in current_setting:
        split_based_on = current_setting['split_based_on']

    eval_split = 'test'
    if 'eval_split' in current_setting:
        eval_split = current_setting['eval_split']

    binarize_labels = False
    if 'binarize_labels' in current_setting:
        binarize_labels = current_setting['binarize_labels']

    model_dtype_setting = 'bf16'
    if 'model_dtype' in current_setting:
        model_dtype_setting = current_setting['model_dtype']
    training_dtype = 'bf16'
    if 'training_dtype' in current_setting:
        training_dtype = current_setting['model_dtype']
    do_quantization_setting = True
    if 'do_quantization' in current_setting:
        do_quantization_setting = current_setting['do_quantization']
    is_classification = True
    if 'do_train_as_classification' in current_setting:
        is_classification = current_setting['do_train_as_classification']
    learning_rate = 3e-4
    if 'learning_rate' in current_setting:
        learning_rate = current_setting['learning_rate']
    train_using_lora = True
    if 'train_using_lora' in current_setting:
        train_using_lora = current_setting['train_using_lora']
    early_stopping_metric = None
    if 'early_stopping_metric' in current_setting:
        early_stopping_metric = current_setting['early_stopping_metric']

    if hf_token_path:
        with open(hf_token_path) as f:
            hf_token = f.read().strip()
            from huggingface_hub import login
            login(token=hf_token, new_session=False, add_to_git_credential=True)

    attributes = current_setting['annotator_attributes']

    output_model_name = to_simplified_output_name(
        seed,
        current_setting['output_model_name'],
        tasks,
        attributes,
        split_based_on,
        learning_rate
    )

    logging.info(f"Loaded setting {setting_id}: {current_setting}")

    model_dir = Path(model_output_dir) / output_model_name

    if model_dir.exists():
        logging.info(f"Model directory exists, skipping...")
        sys.exit(0)

    eval_output_name = to_simplified_output_name(
        seed,
        current_setting['output_model_name'],
        tasks,
        attributes,
        split_based_on,
        learning_rate,
        eval_split=eval_split
    )
    test_data_output_path = Path(f'data/out/{current_setting["output_model_name"]}/{eval_output_name}.csv')
    if test_data_output_path.exists():
        logging.info(f"Evaluation output exists, skipping...")
        sys.exit(0)

    ###########################################################
    # LOAD DATA
    ###########################################################

    splits = ['train', 'val']
    if not save_final_model:
        splits += ['test']

    data = MergedDataset(
        data_path,
        tasks=tasks,
        token_mapping=None, # build from entire dataset
        load_splits=splits,
        split_based_on=split_based_on,
        attributes=attributes,
        do_binarize=binarize_labels,
        as_classification_tasks=is_classification,
        n=n_train
    )

    # convert into HuggingFace datasets
    train_data = data.train
    dev_data = data.val
    logging.info(
        f"Converted TRAIN and DEV data to huggingface datasets: {len(train_data)} and {len(dev_data)} rows"
    )

    ###########################################################
    # TOKENIZE DATA
    ###########################################################

    # max 95 percentile length determined from data exploration (128 tokens, politeness) plus template (10 tokens)
    # for datasets 'intimacy', 'politeness', 'offensiveness', 'diaz'
    # 'dices-990' is much longer with 95 percentile 232 tokens
    # TODO will need to change if template or token concatenation changes
    CUTOFF = 232

    # from data exploration on complete merged dataset + buffer
    # see notebooks/explore/merged.ipynb
    ATTRIBUTE_TEXT_MAX_NUM_TOKENS = {
        'age': 6,
        'education': 7,
        'gender': 3,
        'race': 6,
        'user_id': 7
    }

    def calc_num_attribute_tokens(attributes):
        num_tokens = 0
        for attribute, tokens_config in attributes.items():
            if 'num_tokens' in tokens_config:
                num_tokens += tokens_config['num_tokens'] + 1
            elif 'use_description' in tokens_config:
                num_tokens = ATTRIBUTE_TEXT_MAX_NUM_TOKENS[attribute]
        # double to account for white space between tokens
        num_tokens = num_tokens * 2
        return num_tokens

    num_attribute_tokens = calc_num_attribute_tokens(attributes)

    tokenizer = TokenizerWithSpecialTokens(
        [], # do not add description tokens as special tokens
        base_model, 
        cutoff = num_attribute_tokens + CUTOFF
    )
 
    prompter = Prompter(attribute_token_maps=data.value2token)
    generate_prompt_and_tokenize = AttributePromptPreprocessorForClassification(
        tokenizer,
        prompter,
        label_column="label"
    )
   
    # apply tokenization to TRAIN and DEV data
    train_data = train_data.shuffle(seed=seed).map(generate_prompt_and_tokenize)
    dev_data = dev_data.shuffle(seed=seed).map(generate_prompt_and_tokenize)
    logging.info(f"Tokenized TRAIN and DEV data")

    ###########################################################
    # LOAD MODEL
    ###########################################################

    do_quantization = do_quantization_setting and not use_cpu
    model_dtype = decide_model_dtype(model_dtype_setting, use_cpu)

    if do_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        quantization_config = None
    
    logging.info(f"Quantization config is {quantization_config}")

    model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            num_labels=data.num_labels,
            eos_token_id=tokenizer.tokenizer.eos_token_id,
            bos_token_id=tokenizer.tokenizer.bos_token_id,
            pad_token_id=tokenizer.tokenizer.pad_token_id
    )

    logging.info(f"Loaded base model: {base_model}")

    if train_using_lora:
        if do_quantization:
            #TODO check layer normalization needs to be fp32
            model = prepare_model_for_kbit_training(model)

        # set lora target modules, depending on base model
        if lora_target_modules is None:
            lora_target_modules = "all-linear"
            if "t5" in base_model.lower():
                embedding_layers = ['embed_tokens']
                classification_layers = ['classification_head.dense', 'classification_head.out_proj']
            elif "gpt2" in base_model.lower():
                embedding_layers = ['wte']
                classification_layers = ['score']
            elif "llama" in base_model.lower():
                embedding_layers = ['embed_tokens']
                classification_layers = ['score']
            elif "qwen" in base_model.lower():
                embedding_layers = ['embed_tokens']
                classification_layers = ['score']
            elif "gemma" in base_model.lower():
                embedding_layers = ['embed_tokens']
                classification_layers = ['score']
            elif "mistral" in base_model.lower():
                embedding_layers = ['embed_tokens']
                classification_layers = ['score']
            else:
                raise NotImplementedError(f'Unknown model type for LoRA: {base_model}')

        if "gpt2" in base_model.lower():
            lora_fan_in_fan_out = True
        else:
            lora_fan_in_fan_out = False

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            modules_to_save=embedding_layers+classification_layers,
            fan_in_fan_out=lora_fan_in_fan_out,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS"
        )
        model = get_peft_model(model, config)
        logging.info(f"Converted model to PEFT model with Lora")
        model.print_trainable_parameters()
        logging.info(f"LoRA target modules: {lora_target_modules}")
        logging.info(f"Fine-tuning: {embedding_layers}, {classification_layers}")


    ##################
    # RUN TRAINING
    ##################

    logging.info(
        f"Gradient accumulation steps: {gradient_accumulation_steps} ({batch_size} / {micro_batch_size})"
    )

    do_fp16, do_bf16 = decide_mixed_precision_training(training_dtype, use_cpu)

    # set up training arguments
    t_args_dict = {
        'per_device_train_batch_size':micro_batch_size,
        'gradient_accumulation_steps':gradient_accumulation_steps,
        'warmup_steps':10,
        'num_train_epochs':num_epochs,
        'learning_rate':learning_rate,
        'fp16': do_fp16,
        'bf16': do_bf16,
        'logging_strategy': 'steps',
        'logging_steps':10,
        'optim': "adamw_torch" if use_cpu else "adamw_torch_fused",
        'eval_strategy':"epoch",
        'save_strategy':'epoch' if early_stopping_metric else 'no',
        'output_dir':model_dir,
        'save_total_limit': 10 if train_using_lora else 3,
        'load_best_model_at_end': True if early_stopping_metric else False, 
        'group_by_length':group_by_length,
        'report_to':None,
        'run_name':None,
        'use_cpu':use_cpu,
        'seed':seed
    }

    if early_stopping_metric:
        t_args_dict['metric_for_best_model'] = early_stopping_metric
        t_args_dict['greater_is_better'] = is_greater_better(early_stopping_metric)

    # set up trainer
    data_collator = transformers.DataCollatorWithPadding(
        tokenizer.tokenizer,
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True
    )

    t_args = transformers.TrainingArguments(**t_args_dict)
    trainer_class = transformers.Trainer

    callbacks = [EarlyStoppingCallback(early_stopping_patience=1)] if early_stopping_metric else []

    trainer = trainer_class(
        model=model,
        train_dataset=train_data,
        eval_dataset=dev_data,
        args=t_args,
        data_collator=data_collator,
        compute_metrics= None if is_classification else compute_regression_metrics,
        callbacks=callbacks
    )

    model.config.use_cache = False

    # torch compile model if possible
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # train model
    if early_stopping_metric:
        logging.info(f"Training model for max {num_epochs} epochs with early stopping.")
    else:
        logging.info(f"Training model for {num_epochs} epochs")
    train_output = trainer.train()

    ######
    # SAVE TRAINING METRICS
    #####

    log_history = trainer.state.log_history
    training_logs = [log for log in log_history[:-1] if 'eval_loss' not in log]
    if len(training_logs) > 0:
        df_training_logs = pd.DataFrame(training_logs).set_index('step')
        experiment_config_path = Path(configuration_path)
        log_path = Path('logs/train') / experiment_config_path.parent.name / experiment_config_path.stem / (output_model_name + '.csv')
        log_path.parent.mkdir(exist_ok=True, parents=True)
        df_training_logs.to_csv(log_path)
    
    write_training_output(train_output, configuration_path, setting_id)

    ##################
    # SAVE AND EXPORT MODEL
    ##################

    # save model
    if save_final_model:
        logging.info(f"Saving model to {model_dir}")
        model.save_pretrained(model_dir, save_embedding_layers=True)
    else:
        logging.info(f"Evaluating model on test set, because not saving model for later evaluation")
        test_data = data.test
        test_data = test_data.shuffle(seed=seed).map(generate_prompt_and_tokenize)
        metrics = trainer.evaluate(eval_dataset=test_data)
        experiment_config_path = Path(configuration_path)
        results_path = experiment_config_path.parent / f'{experiment_config_path.stem}.training-eval.csv'
        metrics_df = pd.DataFrame.from_records([metrics])
        metrics_df['setting_index'] = setting_id
        metrics_df = metrics_df.set_index('setting_index')
        logging.info(f"Saving metrics to {results_path}")
        skip_header = results_path.exists()
        metrics_df.to_csv(results_path, mode='a', header= not skip_header)

if __name__ == "__main__":
    st = time.time()
    fire.Fire(train)
    logging.info(f"Total execution time: {time.time() - st:.2f} seconds")
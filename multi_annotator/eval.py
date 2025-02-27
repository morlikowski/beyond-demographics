
import json
import time 
import logging
import os
import shutil
import sys

import fire
import torch

from pathlib import Path
from tqdm import tqdm

from transformers import (
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from peft import PeftModel

from multi_annotator.data import MergedDataset
from multi_annotator.utils import map_letter_labels_to_numeric, task_to_scale_max, to_simplified_output_name

from multi_annotator.preprocessing import (
    TokenizerWithSpecialTokens,
)
from multi_annotator.prompting import Prompter

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
    
def main(
        config_path: str,
        setting_index: int,
        models_dir: str,
        # generate label or select label based on probability
        do_generate: bool = False,
        # misc parameters
        log_level: str = 'info',
        hf_token_path = None,
        remove_checkpoints = False
    ):
    
    ###########################################################
    # LOAD SETTINGS
    ###########################################################

    models_dir = Path(models_dir)

    with open(config_path) as file:
        config = json.load(file)

    defaults = config['defaults']
    settings = config['settings']
    s = defaults
    s.update(settings[setting_index])

    seed = s['seed']
    do_binarize = s['binarize_labels'] 
    is_classification = True
    if 'do_train_as_classification' in s:
        is_classification = s['do_train_as_classification']
    base_model_name_or_path = s["base_model"]
    is_chat = 'instruct' in base_model_name_or_path.lower()
    # https://arxiv.org/abs/2208.07339
    do_quantization_setting = True
    if 'do_quantization' in s:
        do_quantization_setting = s['do_quantization']
    model_dtype = 'bf16'
    if 'model_dtype' in s:
        model_dtype = s['model_dtype']
    trained_using_lora = False
    if 'train_using_lora' in s:
        trained_using_lora = s['train_using_lora']
    split_based_on = 'instance'
    if 'split_based_on' in s:
        split_based_on = s['split_based_on']
    learning_rate = 3e-4
    if 'learning_rate' in s:
        learning_rate = s['learning_rate']
    do_prompt = False
    if 'do_prompt' in s:
        do_prompt = s['do_prompt']
    attribute_format = None
    if 'attribute_format' in s:
        attribute_format = s['attribute_format']

    eval_split = 'test'
    if 'eval_split' in s:
        eval_split = s['eval_split']

    test_data_input_path = s["data_path"]
    n_eval_samples = s['n_eval'] if 'n_eval' in s else -1
    tasks = s['tasks']
    if 'eval_tasks' in s:
        tasks = s['eval_tasks']
    if len(tasks) > 1:
        raise NotImplementedError('Currently only evaluating on a single task is supported. However, only a few changes are needed to support multiple tasks.')
    attributes = s['annotator_attributes']

    trained_model_name = to_simplified_output_name(
        seed,
        s['output_model_name'],
        s['tasks'],
        attributes,
        split_based_on,
        learning_rate,
        is_prompting=do_prompt
    )

    adapter_or_finetuned_path = models_dir / trained_model_name

    # set up logging
    logging.basicConfig(level=getattr(logging, log_level.upper()), format='%(asctime)s %(levelname)s %(message)s')

    if hf_token_path:
        with open(hf_token_path) as f:
            hf_token = f.read().strip()
            from huggingface_hub import login
            login(token=hf_token, new_session=False, add_to_git_credential=True)

    output_name = to_simplified_output_name(
        seed,
        s['output_model_name'],
        tasks,
        attributes,
        split_based_on,
        learning_rate,
        is_prompting=do_prompt,
        eval_split=eval_split
    )

    eval_data_output_path = Path(f'data/out/{s["output_model_name"]}/{output_name}.csv')
    if eval_data_output_path.exists():
        logging.info(f"Evaluation output exists, skipping...")
        sys.exit(0)

    ###########################################################
    # SET UP
    ###########################################################

    # set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Running on device: {device}")
    if device == "cuda":
        logging.info(f"CUDA memory: {round(torch.cuda.mem_get_info()[0]/1024**3,2)}GB")

    # set up methods with progress indicators for pandas
    tqdm.pandas()
    data = MergedDataset(
        test_data_input_path,
        tasks=tasks,
        load_splits=[eval_split], # test or val
        split_based_on=split_based_on,
        token_mapping=None, # build from entire dataset
        attributes=attributes,
        do_binarize=do_binarize,
        as_classification_tasks=is_classification,
        n=n_eval_samples,
        load_as_hf_dataset=False
    )

    ###########################################################
    # LOAD DATA
    ###########################################################

    # load EVAL data
    if eval_split == 'val':
        eval_df = data.val
    elif eval_split == 'test':
        eval_df = data.test
    logging.info(f"Loaded EVAL data: {eval_df.shape[0]} rows")

    # print 3 random texts
    logging.info(f"3 random texts from EVAL data:\n{eval_df.sample(3, random_state=123)['text'].tolist()}\n")

    ###########################################################
    # INIT TOKENIZER
    ###########################################################

    # max length determined from data exploration (253 tokens) plus template (10 tokens)
    # for datasets 'intimacy', 'politeness', 'offensiveness', 'diaz'
    # TODO not true for chat template will not cutoff there as only inference
    # TODO will need to change if template or token concatenation changes
    # TODO need to count descriptions tokens instead of num tokens if using description
    CUTOFF = 270
    # tokens plus attribute seperator (+1)
    num_tokens = sum([settings['num_tokens'] + 1 if 'num_tokens' in settings else 1 for settings in attributes.values()])
    # double to account for white space between tokens
    num_attribute_tokens = num_tokens * 2

    if do_prompt and not is_chat and attribute_format == 'long' and CUTOFF <= 270:
        raise ValueError('If non-chat is used, need to increase cutoff if using long attribute format.')

    tokenizer = TokenizerWithSpecialTokens(
        [], 
        base_model_name_or_path,
        cutoff = num_attribute_tokens + CUTOFF
    )

    ###########################################################
    # LOAD MODEL
    ###########################################################

    logging.info(f"Loading model {base_model_name_or_path}")
    if adapter_or_finetuned_path:
        logging.info(f'{adapter_or_finetuned_path} LoRA weights')
    else:
        logging.info(f'Not using LoRA weights')

    model_dtype = decide_model_dtype(model_dtype, use_cpu = device == 'cpu')

    do_quantization = do_quantization_setting and not device == 'cpu'
    if do_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        quantization_config = None

    if do_prompt:
        if trained_using_lora:
            # adapter-tuned language model
            with open(adapter_or_finetuned_path / Path('adapter_config.json')) as f:
                adapter_config = json.load(f)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                quantization_config=quantization_config,
                torch_dtype=model_dtype,
                device_map=device,
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, adapter_or_finetuned_path)
            model.generation_config.pad_token_id = tokenizer.tokenizer.pad_token_id
            logging.info(f"Also loaded weights for {adapter_config['modules_to_save']}")
        else:
            #un-tuned language model
            model = AutoModelForCausalLM.from_pretrained(
                    base_model_name_or_path,
                    quantization_config=quantization_config,
                    torch_dtype=model_dtype,
                    device_map=device,
                    trust_remote_code=True,
                    eos_token_id=tokenizer.tokenizer.eos_token_id,
                    bos_token_id=tokenizer.tokenizer.bos_token_id,
                    pad_token_id=tokenizer.tokenizer.pad_token_id
                )
            model.generation_config.pad_token_id = tokenizer.tokenizer.pad_token_id
    elif trained_using_lora:
        # adapter-tuned classifier
        with open(adapter_or_finetuned_path / Path('adapter_config.json')) as f:
            adapter_config = json.load(f)
        base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name_or_path,
                quantization_config=quantization_config,
                torch_dtype=model_dtype,
                num_labels=data.num_labels,
                device_map=device,
                trust_remote_code=True
            )
        model = PeftModel.from_pretrained(base_model, adapter_or_finetuned_path)
        logging.info(f"Also loaded weights for {adapter_config['modules_to_save']}")
    else:
        # classifier without adapter-tuning 
        model = AutoModelForSequenceClassification.from_pretrained(
            adapter_or_finetuned_path,
            quantization_config=quantization_config,
            torch_dtype=model_dtype,
            num_labels=data.num_labels,
            device_map=device,
            trust_remote_code=True
        )
    model.eval()

    ###########################################################
    # APPLY TEMPLATE AND PREDICT
    ###########################################################

    prompter = Prompter(
        attribute_token_maps=data.value2token,
        task = tasks[0] if do_prompt else None,
        is_chat = is_chat,
        attribute_format = attribute_format
    )

    def predict(row):
        full_prompt = prompter.apply_prompt(
            row, 
            ignore_task=True # ignore task to use preset template in Prompter which we set to "sequence classification" (task=None)
        )
        tokenized = tokenizer(full_prompt, return_tensors='pt', add_eos_token=False)
        tokenized.to(device)
        with torch.no_grad():
            output = model(**tokenized)
        if is_classification:
            return output.logits.argmax().item()
        else:
            return output.logits.item()
        
    def prompt(row):
        full_prompt = prompter.apply_prompt(row)
        tokenized = tokenizer(full_prompt, return_tensors='pt', add_eos_token=False)
        tokenized.to(device)
        input_len = tokenized['input_ids'][0].shape[0]
        output_ids = model.generate(**tokenized, max_new_tokens=5)
        generated_ids = output_ids[:, input_len:]
        label = tokenizer.tokenizer.batch_decode(generated_ids)[0]
        return label
    
    def batched_prompt(batch, is_chat=False):
        terminators = [
            tokenizer.tokenizer.eos_token_id
        ]
        full_prompts = batch.apply(prompter.apply_prompt, axis=1).tolist()
        if is_chat:
            tokenized = tokenizer.tokenize_chat_prompts(full_prompts, add_eos_token=False)
            if 'llama3' in base_model_name_or_path:
                # recommended in hf tutorial
                terminators.append(tokenizer.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        else:
            tokenized = tokenizer(
                full_prompts,
                return_tensors='pt', 
                add_eos_token=False, 
                padding=True
            )
        tokenized.to(device)
        padded_input_len = tokenized['input_ids'][0].shape[0]
        output_ids = model.generate(**tokenized, max_new_tokens=5, eos_token_id=terminators)
        generated_ids = output_ids[:, padded_input_len:]
        labels = tokenizer.tokenizer.batch_decode(generated_ids)
        return labels
    
    def do_batched_prompt(df, batch_size = 8, is_chat=False):
        num_examples = len(eval_df.index)
        num_batches = num_examples // batch_size
        size_last_batch = num_examples % batch_size
        if size_last_batch > 0:
            num_batches += 1
            logging.info(f"Includes partial last batch of {size_last_batch} example(s)")
        logging.info(f"Calculating labels for {num_batches} batches")
        batch_indecies = zip(range(0, num_examples, batch_size), range(batch_size, num_examples+1, batch_size))
        predictions = []
        generations = []
        with tqdm(total=num_batches) as progress:
            for i, j in batch_indecies:
                batch = df.iloc[i:j]
                generated_texts = batched_prompt(batch, is_chat=is_chat)
                batch_predictions = [map_letter_labels_to_numeric(text, scale_1_to=scale_max) for text in generated_texts]
                predictions.extend(batch_predictions)
                generations.extend(generated_texts)
                progress.update(1)
            if size_last_batch > 0:
                batch = df.iloc[-1 * size_last_batch:]
                generated_texts = batched_prompt(batch, is_chat=is_chat)
                batch_predictions = [map_letter_labels_to_numeric(text, scale_1_to=scale_max) for text in generated_texts]
                predictions.extend(batch_predictions)
                generations.extend(generated_texts)
                progress.update(1)
        return predictions, generations
 
    if do_prompt:
        scale_max = task_to_scale_max(tasks[0])
        predictions, generations = do_batched_prompt(eval_df, is_chat = is_chat)
        eval_df['prediction'] = predictions
        eval_df['generation'] = generations
    else:
        logging.info(f"Calculating labels for {len(eval_df.index)} texts")
        eval_df['prediction'] = eval_df.progress_apply(predict, axis=1)

    eval_data_output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving predictions to {eval_data_output_path}")
    eval_df.to_csv(eval_data_output_path, index=False)

    if remove_checkpoints:
        shutil.rmtree(adapter_or_finetuned_path)
        logging.info(f'Removed directory including subdirectories: {adapter_or_finetuned_path}')

if __name__ == "__main__":
    st = time.time()
    fire.Fire(main)
    logging.info(f'Total execution time: {time.time() - st:.2f} seconds')
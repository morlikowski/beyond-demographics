import logging
from typing import List

from transformers import AutoTokenizer

from multi_annotator.prompting import Prompter
from multi_annotator.utils import label_to_completion

ADD_EOS_TOKEN = False
TRAIN_ON_INPUTS = True

class TokenizerWithSpecialTokens():
    def __init__(self, 
                 special_tokens: List[str], 
                 base_model: str, 
                 cutoff: int = 260
        ):
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=False,
            split_special_tokens=False,
            additional_special_tokens=special_tokens,
            padding_side='left'
        )

        # fix tokenizer if it doesn't have a pad token
        if not getattr(tokenizer, "pad_token", None):
            logging.warning(
                "Couldn't find a PAD token in the tokenizer, using the EOS token instead."
            )
            tokenizer.pad_token = tokenizer.eos_token
        
        self.tokenizer = tokenizer
        self.cutoff = cutoff
        self.base_model = base_model
    
    def __call__(self, text, add_eos_token=True, return_tensors=None, padding=False):
        result = self.tokenizer(
            text,
            truncation=True,
            max_length=self.cutoff,
            padding=padding,
            return_tensors=return_tensors,
        )
        if (
            add_eos_token
            and result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        # TODO double check this
        """
        in both LlamaForCausalLM and FalconForCausalLM, logits are shifted
        see also: https://github.com/tloen/alpaca-lora/issues/365
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        """

        return result
    
    def tokenize_chat_prompts(self, full_prompts: List[List[dict]], do_tokenize=True, add_eos_token=False):

        if not 'llama-3' in self.base_model.lower():
            raise NotImplementedError('Chat template post-processing makes assumptions about chat template. Currently only tested with Llama3-Instruct')

        tokenized = self.tokenizer.apply_chat_template(
                full_prompts,
                tokenize=do_tokenize,
                add_generation_prompt=False, # Prompter class already adds start of assistant turn
                continue_final_message=not add_eos_token,
                return_tensors="pt",
                return_dict=do_tokenize,
                padding=True
            )

        return tokenized
    
    def __len__(self):
        return len(self.tokenizer)
        
class AttributePromptPreprocessor():

    def __init__(self, 
                 tokenizer, 
                 prompter: Prompter,
                 label_column = "", 
                 add_label_to_input=False,
                 input_as_label=True, # autoregressive (True) or seq2seq (False)
                 is_chat = False
            ):
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.add_label = add_label_to_input
        self.label_column = label_column
        self.input_as_label = input_as_label
        self.is_chat = is_chat

    def _row_to_label(self, data_point):
        label_col = self.label_column
        task = data_point['task']
        label = data_point[label_col]
        completion = label_to_completion(task, label)
        return completion
        
        
    def __call__(self, data_point):

        label = self._row_to_label(data_point)

        prompt = self.prompter.apply_prompt(
            data_point
        )

        if self.add_label:
            if self.is_chat:
                ASSISTANT_TURN = -1
                partial_answer = prompt[ASSISTANT_TURN]['content']
                prompt[ASSISTANT_TURN]['content'] = f"{partial_answer}{label}"
            else:
                prompt = f"{prompt}{label}"

        if self.is_chat:
            #NOTE as not experimenting with instruction-tuning on DICES, no need for cut-off:
            #longest example in Politeness (205 tokens) is still shorter than 95 percentile in DICES 990 (224 tokens)
            #TODO if needed: cut off example content (based on characters), do not cut completion with label
            tokenized_prompt = self.tokenizer.tokenize_chat_prompts(
                [prompt],
                # when we train on chat completions, the input contains the complete text and should be marked with eot
                add_eos_token=self.add_label
            )
            # apply_chat_prompt tokenization adds an extra dimension
            tokenized_prompt['input_ids'] = tokenized_prompt['input_ids'].squeeze()
            tokenized_prompt['attention_mask'] = tokenized_prompt['attention_mask'].squeeze()
        else:
            tokenized_prompt = self.tokenizer(prompt, add_eos_token=self.add_label)

        if not self.add_label:
            if self.input_as_label:
                completed_prompt = f"{prompt}{label}"
                tokenized_completed_prompt = self.tokenizer(completed_prompt)
                tokenized_prompt["labels"] = tokenized_completed_prompt["input_ids"]
            else:
                label_tokenized = self.tokenizer(label)
                tokenized_prompt["labels"] = label_tokenized['input_ids']

        # we can use this to mask the instruction if we don't want to train on that
        if not TRAIN_ON_INPUTS:
            user_prompt = self.prompter.generate_prompt(data_point['text'])
            tokenized_user_prompt = self.tokenizer(
                user_prompt, add_eos_token=ADD_EOS_TOKEN)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if ADD_EOS_TOKEN:
                user_prompt_len -= 1
            tokenized_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_prompt["labels"][user_prompt_len:]

        return tokenized_prompt
    
class AttributePromptPreprocessorForClassification():

    def __init__(self, tokenizer, prompter: Prompter, label_column = ""):
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.label_column = label_column

    def __call__(self, data_point):

        prompt = self.prompter.apply_prompt(
            data_point,
            ignore_task=True # ignore task to use preset template in prompter which we set to "sequence classification"
        )

        # LlamaForSequenceClassification predicts from the last token that is not a padding token
        # since we are using the EOS token as padding token it would be ignored here anyway
        # so we are not adding the EOS token
        # https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaForSequenceClassification
        tokenized_prompt = self.tokenizer(prompt, add_eos_token=False)

        label_col = self.label_column
        tokenized_prompt["labels"] = data_point[label_col]
        
        # we can use this to mask the instruction if we don't want to train on that
        if not TRAIN_ON_INPUTS:
            user_prompt = self.prompter.generate_prompt(data_point['text'])
            tokenized_user_prompt = self.tokenizer(
                user_prompt, add_eos_token=ADD_EOS_TOKEN)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if ADD_EOS_TOKEN:
                user_prompt_len -= 1
            tokenized_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_prompt["labels"][user_prompt_len:]

        return tokenized_prompt
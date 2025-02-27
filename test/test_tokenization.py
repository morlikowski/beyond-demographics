import pytest
from multi_annotator.prompting import Prompter
from multi_annotator.data import VALUE2DESC
from multi_annotator.preprocessing import TokenizerWithSpecialTokens

@pytest.fixture
def tokenizer():
    return TokenizerWithSpecialTokens(
        [], 
        'meta-llama/Meta-Llama-3-8B-Instruct',
        cutoff = 270
    )

def test_chat_template(tokenizer):
    data_point = {
        'gender': 'Woman',
        'age': '40-44',
        'education': 'College degree',
        'race': 'Arab',
        'text': 'Just an example text!',
        'task': 'diaz'
    }

    prompter = Prompter({}, task='diaz', is_chat=True, attribute_format='long')
    prompt = prompter.apply_prompt(data_point)

    chat_templated = tokenizer.tokenize_chat_prompts(
        [prompt], 
        do_tokenize=False
    )

    expected_string = (
        '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
        '**Question**: Consider you read this text, what do you think is the sentiment it expresses?\n'
        '**Text**: Just an example text!\n'
        '(A) very negative\n(B) somewhat negative\n(C) neutral\n(D) somewhat positive\n(E) very positive<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
        '**Answer**: ('
    )
    assert chat_templated[0] == expected_string

def test_chat_template_tokenized(tokenizer):
    data_point = {
        'gender': 'Woman',
        'age': '40-44',
        'education': 'College degree',
        'race': 'Arab',
        'text': 'Just an example text!',
        'task': 'diaz'
    }

    prompter = Prompter({}, task='diaz', is_chat=True, attribute_format='long')
    prompt = prompter.apply_prompt(data_point)

    chat_tokenized = tokenizer.tokenize_chat_prompts(
        [prompt], 
        do_tokenize=True
    )

    expected_token_ids = [128000, 128006,    882, 128007,    271,    334,  14924,  96618,  21829,
           499,   1373,    420,   1495,     11,   1148,    656,    499,   1781,
           374,    279,  27065,    433,  61120,   5380,    334,   1199,  96618,
          4702,    459,   3187,   1495,   4999,   4444,      8,  15668,   8389,
           198,   5462,      8,  18024,  81575,   8389,    198,   3100,      8,
         59794,    198,   5549,      8,  18024,  81575,   6928,    198,  10953,
             8,  15668,   6928, 128009, 128006,  78191, 128007,    271,    334,
         16533,  96618,    320]

    assert chat_tokenized['input_ids'][0].tolist() == expected_token_ids
from multi_annotator.prompting import Prompter
from multi_annotator.data import VALUE2TOKEN
from multi_annotator.data import VALUE2DESC

def test_apply_prompt_sequence_classification_textual_attributes():
    data_point = {
        'gender': 'Woman',
        'age': '40-44',
        'education': 'College degree',
        'race': 'Arab',
        'text': 'Just an example text!',
        'task': 'diaz'
    }
    prompter = Prompter(VALUE2DESC)
    prompt = prompter.apply_prompt(data_point, ignore_task=True)
    assert prompt == 'Annotator: arab, 40 to 44 years old, a woman, a college degree\nText: Just an example text!'

def test_apply_prompt_sequence_classification_content_only():
    data_point = {
        'gender': 'Woman',
        'age': '40-44',
        'education': 'College degree',
        'race': 'Arab',
        'text': 'Just an example text!',
        'task': 'diaz'
    }
    prompter = Prompter(attribute_token_maps={})
    prompt = prompter.apply_prompt(data_point, ignore_task=True)
    assert prompt == 'Just an example text!'

def test_apply_prompt_chat_default_format_textual_attributes():
    data_point = {
        'gender': 'Woman',
        'age': '40-44',
        'education': 'College degree',
        'race': 'Arab',
        'text': 'Just an example text!',
        'task': 'diaz'
    }
    prompter = Prompter(VALUE2DESC, task='diaz', is_chat=True)
    prompt = prompter.apply_prompt(data_point)
    assert prompt[0]['content'] == 'You are arab, 40 to 44 years old, a woman, a college degree\n'

def test_apply_prompt_chat_long_format_textual_attributes():
    data_point = {
        'gender': 'Woman',
        'age': '40-44',
        'education': 'College degree',
        'race': 'Arab',
        'text': 'Just an example text!',
        'task': 'diaz'
    }
    prompter = Prompter(VALUE2DESC, task='diaz', is_chat=True, attribute_format='long')
    prompt = prompter.apply_prompt(data_point)
    assert prompt[0]['content'] == 'In terms of race or ethnicity, you are arab. You are 40 to 44 years old. In terms of gender, you are a woman. The highest degree or level of school that you have completed is a college degree.\n'

def test_apply_prompt_chat_content_only():
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
    assert prompt[0]['content'] == '**Question**: Consider you read this text, what do you think is the sentiment it expresses?\n**Text**: Just an example text!\n(A) very negative\n(B) somewhat negative\n(C) neutral\n(D) somewhat positive\n(E) very positive\n'

def test_apply_prompt_sequence_classification_annotator_id():
    data_point = {
        'user_id': 'diaz_R_Xi7Yzei0BplKWLD',
        'text': 'Just an example text!',
        'task': 'diaz'
    }
    token_map = {
        'user_id': {
            'diaz_R_Xi7Yzei0BplKWLD': ['42']
        }
    }

    prompter = Prompter(token_map)
    prompt = prompter.apply_prompt(data_point, ignore_task=True)
    assert prompt == 'Annotator: unique identifier 42\nText: Just an example text!'

def test_apply_prompt_chat_annotator_id():
    data_point = {
        'user_id': 'diaz_R_Xi7Yzei0BplKWLD',
        'text': 'Just an example text!',
        'task': 'diaz'
    }
    token_map = {
        'user_id': {
            'diaz_R_Xi7Yzei0BplKWLD': ['42']
        }
    }

    prompter = Prompter(token_map, task='diaz', is_chat=True)
    prompt = prompter.apply_prompt(data_point)
    assert prompt[0]['content'] == 'You have the unique identifier 42\n'

def test_apply_prompt_sequence_classification_annotator_id_plus_attributes():
    data_point = {
        'user_id': 'diaz_R_Xi7Yzei0BplKWLD',
        'text': 'Just an example text!',
        'gender': 'Woman',
        'age': '40-44',
        'education': 'College degree',
        'race': 'Arab',
        'task': 'diaz'
    }
    token_map = {
        'user_id': {
            'diaz_R_Xi7Yzei0BplKWLD': ['42']
        }
    }
    token_map.update(VALUE2DESC)

    prompter = Prompter(token_map)
    prompt = prompter.apply_prompt(data_point, ignore_task=True)
    assert prompt == 'Annotator: unique identifier 42, arab, 40 to 44 years old, a woman, a college degree\nText: Just an example text!'

def test_apply_prompt_chat_long_format_annotator_id_plus_attributes():
    data_point = {
        'user_id': 'diaz_R_Xi7Yzei0BplKWLD',
        'text': 'Just an example text!',
        'gender': 'Woman',
        'age': '40-44',
        'education': 'College degree',
        'race': 'Arab',
        'task': 'diaz'
    }
    token_map = {
        'user_id': {
            'diaz_R_Xi7Yzei0BplKWLD': ['42']
        }
    }
    token_map.update(VALUE2DESC)

    prompter = Prompter(token_map, task='diaz', is_chat=True, attribute_format='long')
    prompt = prompter.apply_prompt(data_point)
    assert prompt[0]['content'] == 'You have the unique identifier 42. In terms of race or ethnicity, you are arab. You are 40 to 44 years old. In terms of gender, you are a woman. The highest degree or level of school that you have completed is a college degree.\n'

def test_apply_prompt_chat_annotator_id_plus_attributes():
    data_point = {
        'user_id': 'diaz_R_Xi7Yzei0BplKWLD',
        'text': 'Just an example text!',
        'gender': 'Woman',
        'age': '40-44',
        'education': 'College degree',
        'race': 'Arab',
        'task': 'diaz'
    }
    token_map = {
        'user_id': {
            'diaz_R_Xi7Yzei0BplKWLD': ['42']
        }
    }
    token_map.update(VALUE2DESC)

    prompter = Prompter(token_map, task='diaz', is_chat=True)
    prompt = prompter.apply_prompt(data_point)
    assert prompt[0]['content'] == 'You have the unique identifier 42. You are arab, 40 to 44 years old, a woman, a college degree\n'

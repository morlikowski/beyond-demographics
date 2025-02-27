from multi_annotator.utils import to_simplified_output_name
from multi_annotator.utils import map_letter_labels_to_numeric


def test_attributes_plus_id_to_simplified_output_name():
    attributes = {
        'user_id': {'use_description': True}, 
        'race': {'use_description': True}, 
        'gender': {'use_description': True}, 
        'age': {'use_description': True}, 
        'education': {'use_description': True}
    }
    output_name = to_simplified_output_name(
        seed=459293949, 
        output_model_name='llama3', 
        tasks=['politeness'], 
        attributes=attributes, 
        split_based_on='instance', 
        lr=6e-05, 
        is_prompting=False, 
        eval_split='eval'
    )
    assert output_name == '459293949-llama3-politeness-id+textual-instance-6e-05'

def test_map_letter_labels_to_numeric_1_to_5():
    A = map_letter_labels_to_numeric('A', scale_1_to = 5)
    assert A == 0.0
    B = map_letter_labels_to_numeric('B', scale_1_to = 5)
    assert B == 0.25
    C = map_letter_labels_to_numeric('C', scale_1_to = 5)
    assert C == 0.5
    D = map_letter_labels_to_numeric('D', scale_1_to = 5)
    assert D == 0.75
    E = map_letter_labels_to_numeric('E', scale_1_to = 5)
    assert E == 1.0

def test_map_letter_labels_to_numeric_1_to_3():
    A = map_letter_labels_to_numeric('A', scale_1_to = 3)
    assert A == 0.0
    B = map_letter_labels_to_numeric('B', scale_1_to = 3)
    assert B == 0.5
    C = map_letter_labels_to_numeric('C', scale_1_to = 3)
    assert C == 1.0
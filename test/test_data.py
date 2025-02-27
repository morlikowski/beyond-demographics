from multi_annotator.data import MergedDataset

def test_load_as_classification_task():
    tasks = [
        ('intimacy', 5, [0, 1, 2, 3, 4]),
        ('politeness', 5, [0, 1, 2, 3, 4]),
        ('offensiveness', 5, [0, 1, 2, 3, 4]),
        ('diaz', 5, [0, 1, 2, 3, 4]),
        ('dices-990', 3, [0, 1, 2])
    ]

    for task, expected_num_labels, expected_label_values in tasks:
        data = MergedDataset(
            'shared_data/processed/merged_data.csv',
            tasks=[task],
            load_splits=['val'],
            split_based_on='instance',
            token_mapping=None,
            attributes={},
            do_binarize=False,
            as_classification_tasks=True,
            load_as_hf_dataset=False
        )
        assert data.num_labels == expected_num_labels
        df = data.val
        assert df['label'].dtype == 'int64'
        # label values are 0 to num_labels - 1
        assert all(df['label'] >= 0)
        assert all(df['label'] < data.num_labels)
        label_values = df['label'].unique()
        assert len(set(label_values).symmetric_difference(set(expected_label_values))) == 0

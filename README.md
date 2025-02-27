# Beyond Demographics (ARR Dec'24)

This repository contains code and data for "Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals’ Subjective Text Perceptions". If any content in this repository is useful for your work, please cite our paper.

## DEMO dataset

The DEMO dataset is a curated and further unified collection of existing datasets. Details are reported in the paper (Link). DEMO is distributed as a CSV file in `shared_data/processed/merged_data.csv`. Code to load DEMO as we do in our experiments is contained in `multi_annotator/data.py`. **If you use DEMO in your work, please cite our paper *as well as the individual datasets*.** Below we include information on the original dataset for each task included in DEMO.

- *Intimacy*
    - tba
- *Politeness*
    - tba
- *Offensiveness*
    - tba
- *Safety*
    - Dataset: [DICES-350](https://github.com/google-research-datasets/dices-dataset/tree/b5596d9edd585361967991a591abc7b11fb1f7a3/350)
    - License: [Creative Commons Attribution 4.0 International License](https://github.com/google-research-datasets/dices-dataset?tab=readme-ov-file#license)
    - Paper: [DICES Dataset: Diversity in Conversational AI Evaluation for Safety](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a74b697bce4cac6c91896372abaa8863-Abstract-Datasets_and_Benchmarks.html)
- *Sentiment*
    - Dataset: [Age Bias Training and Testing Data](https://doi.org/10.7910/DVN/F6EMTS), [Older Adult Annotator Demographic and Attitudinal Survey](https://doi.org/10.7910/DVN/GXS7DI)
    - License: [Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
    - Paper: [Addressing Age-Related Bias in Sentiment Analysis](https://doi.org/10.1145/3173574.3173986)

## Setup

Dependencies are listed in `pyproject.toml` with optional dependencies for `[gpu]` and `[development]`. For local development on the CPU install the PyTorch CPU variant using `requirements.cpu.txt`.
# Beyond Demographics (ARR Dec'24)

This repository contains code and data for "Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals’ Subjective Text Perceptions". If any content in this repository is useful for your work, please cite our paper (available as [preprint on arXiv](https://arxiv.org/abs/2502.20897)).

```
@misc{orlikowski2025demographicsfinetuninglargelanguage,
      title={Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions}, 
      author={Matthias Orlikowski and Jiaxin Pei and Paul Röttger and Philipp Cimiano and David Jurgens and Dirk Hovy},
      year={2025},
      eprint={2502.20897},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.20897}, 
}
```

## DEMO dataset

The DEMO dataset is a curated and further unified collection of existing datasets. Details are reported in the paper ([Link](https://arxiv.org/abs/2502.20897)). DEMO is distributed as a CSV file in `shared_data/processed/merged_data.csv`. Code to load DEMO as we do in our experiments is contained in `multi_annotator/data.py`. **If you use DEMO in your work, please cite our paper *as well as the individual datasets*.** Below we include information on the original dataset for each task included in DEMO.

- *Intimacy*
    - Dataset: [Multilingual Tweet Intimacy Analysis](https://codalab.lisn.upsaclay.fr/competitions/7096#learn_the_details)
    - Paper: [SemEval 2023 Task 9: Multilingual Tweet Intimacy Analysis](https://arxiv.org/abs/2210.01108)
- *Politeness*
    - Dataset: [POPQUORN - Politeness](https://github.com/Jiaxin-Pei/Potato-Prolific-Dataset/tree/main/dataset/politeness_rating)
    - Paper: [When Do Annotator Demographics Matter? Measuring the Influence of Annotator Demographics with the POPQUORN Dataset](https://arxiv.org/abs/2306.06826)
- *Offensiveness*
    - Dataset: [POPQUORN - Offensiveness](https://github.com/Jiaxin-Pei/Potato-Prolific-Dataset/tree/main/dataset/offensiveness)
    - Paper: [When Do Annotator Demographics Matter? Measuring the Influence of Annotator Demographics with the POPQUORN Dataset](https://arxiv.org/abs/2306.06826)
- *Safety*
    - Dataset: [DICES-350](https://github.com/google-research-datasets/dices-dataset/tree/b5596d9edd585361967991a591abc7b11fb1f7a3/350)
    - Paper: [DICES Dataset: Diversity in Conversational AI Evaluation for Safety](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a74b697bce4cac6c91896372abaa8863-Abstract-Datasets_and_Benchmarks.html)
    - License: [Creative Commons Attribution 4.0 International License](https://github.com/google-research-datasets/dices-dataset?tab=readme-ov-file#license)
- *Sentiment*
    - Dataset: [Age Bias Training and Testing Data](https://doi.org/10.7910/DVN/F6EMTS), [Older Adult Annotator Demographic and Attitudinal Survey](https://doi.org/10.7910/DVN/GXS7DI)
    - Paper: [Addressing Age-Related Bias in Sentiment Analysis](https://doi.org/10.1145/3173574.3173986)
    - License: [Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

## Setup

Dependencies are listed in `pyproject.toml` with optional dependencies for `[gpu]` and `[development]`. For local development on the CPU install the PyTorch CPU variant using `requirements.cpu.txt`.

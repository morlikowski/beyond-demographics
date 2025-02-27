#!/bin/bash

sbatch experiments/classification/llama3_classification-notokens-offensiveness-user-lr8e5.sbatch
sbatch experiments/classification/llama3_classification-notokens-politeness-instance-lr8e5.sbatch
sbatch experiments/classification/llama3_classification-textual-politeness-instance-lr8e5.sbatch
sbatch experiments/classification/llama3_classification-textual-diaz-instance-lr6e5.sbatch
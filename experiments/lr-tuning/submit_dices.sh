#!/bin/bash
for FILE in experiments/lr-tuning/llama3_classification-notokens-dices-*.sbatch; do
    sbatch $FILE
    sleep 1
done
for FILE in experiments/lr-tuning/llama3_classification-textual-dices-*.sbatch; do
    sbatch $FILE
    sleep 1
done
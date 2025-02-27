#!/bin/bash
for FILE in experiments/annotator-id/llama3_classification-textual_id-dices-*.sbatch; do
    sbatch $FILE
    sleep 1
done
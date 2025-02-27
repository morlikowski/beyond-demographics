#!/bin/bash
for FILE in experiments/annotator-id-attributes/llama3_classification-id+textual-dices-*.sbatch; do
    sbatch $FILE
    sleep 1
done

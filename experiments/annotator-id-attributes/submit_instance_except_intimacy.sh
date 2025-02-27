#!/bin/bash
for FILE in experiments/annotator-id-attributes/llama3_classification-id+textual-offensiveness-instance*.sbatch; do
    sbatch $FILE
    sleep 1
done
for FILE in experiments/annotator-id-attributes/llama3_classification-id+textual-politeness-instance*.sbatch; do
    sbatch $FILE
    sleep 1
done
for FILE in experiments/annotator-id-attributes/llama3_classification-id+textual-diaz-instance*.sbatch; do
    sbatch $FILE
    sleep 1
done
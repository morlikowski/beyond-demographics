#!/bin/bash
for FILE in experiments/annotator-id-attributes/llama3_classification-id+textual-offensiveness-user*.sbatch; do
    sbatch $FILE
    sleep 1
done
for FILE in experiments/annotator-id-attributes/llama3_classification-id+textual-politeness-user*.sbatch; do
    sbatch $FILE
    sleep 1
done
for FILE in experiments/annotator-id-attributes/llama3_classification-id+textual-diaz-user*.sbatch; do
    sbatch $FILE
    sleep 1
done
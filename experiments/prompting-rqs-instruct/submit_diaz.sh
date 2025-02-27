#!/bin/bash
for FILE in experiments/prompting-rqs-instruct/llama3_instruct*-diaz-*.sbatch; do
    sbatch $FILE
    sleep 1
done
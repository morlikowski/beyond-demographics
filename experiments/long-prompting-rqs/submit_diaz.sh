#!/bin/bash
for FILE in experiments/long-prompting-rqs/llama3_instruct*-diaz-*.sbatch; do
    sbatch $FILE
    sleep 1
done
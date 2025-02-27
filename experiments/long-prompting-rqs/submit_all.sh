#!/bin/bash
for FILE in experiments/long-prompting-rqs/*.sbatch; do
    sbatch $FILE
    sleep 1
done
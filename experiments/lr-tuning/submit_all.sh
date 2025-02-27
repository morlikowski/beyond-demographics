#!/bin/bash
for FILE in experiments/lr-tuning/*.sbatch; do
    sbatch $FILE
    sleep 10
done

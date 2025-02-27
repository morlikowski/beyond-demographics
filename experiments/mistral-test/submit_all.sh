#!/bin/bash
for FILE in experiments/mistral-test/*.sbatch; do
    sbatch $FILE
    sleep 1
done

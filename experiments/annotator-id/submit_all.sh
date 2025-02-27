#!/bin/bash
for FILE in experiments/annotator-id/*.sbatch; do
    sbatch $FILE
    sleep 1
done

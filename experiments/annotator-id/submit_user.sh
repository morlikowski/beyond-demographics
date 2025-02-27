#!/bin/bash
for FILE in experiments/annotator-id/*-user-*.sbatch; do
    sbatch $FILE
    sleep 1
done

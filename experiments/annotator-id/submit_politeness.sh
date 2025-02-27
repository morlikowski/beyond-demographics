#!/bin/bash
for FILE in experiments/annotator-id/*-politeness-*.sbatch; do
    sbatch $FILE
    sleep 1
done

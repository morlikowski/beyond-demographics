#!/bin/bash
for FILE in experiments/annotator-id-test/*-dices-*.sbatch; do
    sbatch $FILE
    sleep 1
done

#!/bin/bash
for FILE in experiments/classification/*-dices-*.sbatch; do
    sbatch $FILE
    sleep 1
done

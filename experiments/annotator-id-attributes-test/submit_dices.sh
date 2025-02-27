#!/bin/bash
for FILE in experiments/annotator-id-attributes-test/*-dices-*.sbatch; do
    sbatch $FILE
    sleep 1
done

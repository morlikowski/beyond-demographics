#!/bin/bash
for FILE in experiments/annotator-id-test/*.sbatch; do
    sbatch $FILE
    sleep 1
done

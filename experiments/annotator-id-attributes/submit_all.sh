#!/bin/bash
for FILE in experiments/annotator-id-attributes/*.sbatch; do
    sbatch $FILE
    sleep 1
done

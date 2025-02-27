#!/bin/bash
for FILE in experiments/annotator-id-attributes-test/*.sbatch; do
    sbatch $FILE
    sleep 1
done

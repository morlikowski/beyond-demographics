#!/bin/bash
for FILE in experiments/classification/*user-lr*.sbatch; do
    echo $FILE
    sbatch $FILE
    sleep 1
done
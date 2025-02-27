#!/bin/bash
for FILE in experiments/classification/*.sbatch; do
    sbatch $FILE
    sleep 1
done

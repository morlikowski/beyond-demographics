#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00
#SBATCH --exclude worker-10
if [[ -n "$SLURM_JOB_ID" ]]; then
    source /homes/morlikowski/multi-annotator/.venv/bin/activate
fi

for config_path in experiments/annotator-id-attributes/*.json; 
do
    echo Computing metrics for $config_path
    path_no_ending=$(echo $config_path | cut -d\. -f1)
    python3 -m multi_annotator.metrics $config_path $path_no_ending.results.val.csv --do_calculate_cis=True
done

if [[ -n "$SLURM_JOB_ID" ]]; then
    deactivate
fi

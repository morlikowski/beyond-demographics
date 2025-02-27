#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00
#SBATCH --exclude worker-10
if [[ -n "$SLURM_JOB_ID" ]]; then
    source /homes/morlikowski/multi-annotator/.venv/bin/activate
fi

config_names="dices intimacy politeness diaz offensiveness"

for name in $config_names; 
do
    echo Computing metrics for llama3_instruct_70b-$name-instance.json
    python3 -m multi_annotator.metrics experiments/prompting-rqs-instruct-70b/llama3_instruct_70b-notokens-prompt-$name-instance.json experiments/prompting-rqs-instruct-70b/llama3_instruct_70b-notokens-prompt-$name-instance.results.csv --is_discrete=True --do_calculate_cis=True
done

if [[ -n $SLURM_JOB_ID ]]; then
    deactivate
fi

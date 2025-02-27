#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00
#SBATCH --exclude worker-10
if [[ -n "$SLURM_JOB_ID" ]]; then
    source /homes/morlikowski/multi-annotator/.venv/bin/activate
fi

config_names="textual-prompt-diaz textual-prompt-intimacy textual-prompt-politeness textual-prompt-offensiveness textual-prompt-dices"

for name in $config_names; 
do
    echo Computing metrics for llama3_instruct_long-$name-user.json
    python3 -m multi_annotator.metrics experiments/long-prompting-rqs/llama3_instruct_long-$name-user.json experiments/long-prompting-rqs/llama3_instruct_long-$name-user.results.csv --is_discrete=True --do_calculate_cis=True
done


if [[ -n $SLURM_JOB_ID ]]; then
    deactivate
fi

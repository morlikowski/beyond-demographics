#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00
#SBATCH --exclude worker-10
if [[ -n "$SLURM_JOB_ID" ]]; then
    source /homes/morlikowski/multi-annotator/.venv/bin/activate
fi

config_names="notokens-prompt-diaz textual-prompt-diaz notokens-prompt-intimacy textual-prompt-intimacy notokens-prompt-politeness textual-prompt-politeness notokens-prompt-offensiveness textual-prompt-offensiveness notokens-prompt-dices textual-prompt-dices"

for name in $config_names; 
do
    echo Computing metrics for llama3_instruct-$name-user.json
    python3 -m multi_annotator.metrics experiments/prompting-rqs-instruct/llama3_instruct-$name-user.json experiments/prompting-rqs-instruct/llama3_instruct-$name-user.results.csv --is_discrete=True
done

if [[ -n $SLURM_JOB_ID ]]; then
    deactivate
fi

#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=5-00
#SBATCH --exclude worker-10
#SBATCH --tmp=35G
if [[ -n "$SLURM_JOB_ID" ]]; then
    source /homes/morlikowski/multi-annotator/.venv/bin/activate
    export HF_HOME=$SLURM_JOB_TMP
fi

export TOKENIZERS_PARALLELISM=true

CONFIG=experiments/classification/llama3_classification-notokens-dices-instance-lr3e5.json
NUM_SETTINGS=$(cat $CONFIG | python3 -c "import json, sys; print(len(json.load(sys.stdin)['settings']))")
let "MAX_INDEX = $NUM_SETTINGS - 1"
echo "Running settings 0 to $MAX_INDEX from $CONFIG"

for i in $(seq 0 $MAX_INDEX);
do
    python3 -m multi_annotator.finetune_cls $CONFIG $i models -hf_token_path=hf_token.txt -micro_batch_size=4 -batch_size=16
    python3 -m multi_annotator.eval $CONFIG $i models --remove_checkpoints -hf_token_path=hf_token.txt
done

if [[ -n $SLURM_JOB_ID ]]; then
    deactivate
fi

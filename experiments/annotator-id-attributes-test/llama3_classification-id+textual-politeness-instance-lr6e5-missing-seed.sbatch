#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00
#SBATCH --exclude worker-10
#SBATCH --tmp=35G
if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "Start to copy virtual env archive"
    rsync -ah --progress /homes/morlikowski/multi-annotator/venv.tar.bz2 $SLURM_JOB_TMP/venv.tar.bz2
    echo "Unpacking virtual env archive"
    tar -xjf $SLURM_JOB_TMP/venv.tar.bz2 -C $SLURM_JOB_TMP/
    echo "Activating virtual env"
    source $SLURM_JOB_TMP/.venv/bin/activate
    export HF_HOME=$SLURM_JOB_TMP
fi

export TOKENIZERS_PARALLELISM=true

CONFIG=experiments/annotator-id-attributes-test/llama3_classification-id+textual-politeness-instance-lr6e5.json
NUM_SETTINGS=$(cat $CONFIG | python3 -c "import json, sys; print(len(json.load(sys.stdin)['settings']))")
echo "Running setting 27 from $CONFIG"

python3 -m multi_annotator.finetune_cls $CONFIG 27 models -hf_token_path=hf_token.txt -micro_batch_size=8
python3 -m multi_annotator.eval $CONFIG 27 models --remove_checkpoints -hf_token_path=hf_token.txt

if [[ -n $SLURM_JOB_ID ]]; then
    deactivate
fi

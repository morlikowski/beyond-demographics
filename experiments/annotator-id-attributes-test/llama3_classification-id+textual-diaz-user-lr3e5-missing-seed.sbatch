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

CONFIG=experiments/annotator-id-attributes-test/llama3_classification-id+textual-diaz-user-lr3e5.json
echo "Running setting 19 from $CONFIG"

python3 -m multi_annotator.finetune_cls $CONFIG 19 models -hf_token_path=hf_token.txt -micro_batch_size=16
python3 -m multi_annotator.eval $CONFIG 19 models --remove_checkpoints -hf_token_path=hf_token.txt

if [[ -n $SLURM_JOB_ID ]]; then
    deactivate
fi

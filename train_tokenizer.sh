#!/bin/bash
#SBATCH --job-name=tokenizer-training
#SBATCH --time=20:00:00
#SBATCH --partition=h100
#SBATCH --nodes 1
#SBATCH --gpus-per-node 8
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.out

set -e
source .env/bin/activate

# get config from arguments
CONFIG=$1
echo "CONFIG: $CONFIG"

# slurm job id
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
PORT_NUM=$((SLURM_JOB_ID + 9999))

# get GPU IDs from slurm (like 4,5)
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS}"
# count the number of GPUs by parsing SLURM_JOB_GPUS (at max 8 GPUs)
IFS=',' read -r -a gpu_array <<< "${SLURM_JOB_GPUS}"
NUM_GPUS=${#gpu_array[@]}
echo "NUM_GPUS: ${NUM_GPUS}"

TORCH_DISTRIBUTED_DEBUG=DETAIL WANDB_MODE=offline accelerate launch --mixed_precision=bf16 --num_machines=1 --num_processes=${NUM_GPUS} --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=${PORT_NUM} --same_network scripts/train_tokenizer.py config=${CONFIG}

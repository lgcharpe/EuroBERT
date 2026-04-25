#!/bin/bash
#SBATCH --account=project_465002893
#SBATCH --partition=standard-g
#SBATCH --nodes=18
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00

#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# set -euo pipefail

mkdir -p slurm_logs

module purge
module use /appl/local/laifs/modules 
module load lumi-aif-singularity-bindings
export HF_HOME=/scratch/project_465002893/nb-embed/huggingface_cache/
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
export SINGULARITYENV_PREPEND_PATH=/user-software/bin

# set MIOPEN temp folder
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

# Tell RCCL to use Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

#Debug
# export NCCL_DEBUG=INFO

# Required on many Cray Slingshot systems for RCCL over CXI
export FI_CXI_ATS=0
export NCCL_CROSS_NIC=1

# Disable GPU Direct RDMA if it's failing silently during channel setup
export NCCL_NET_GDR_LEVEL=0

# For debugging — keep NCCL_DEBUG=INFO and add:
# export NCCL_DEBUG_SUBSYS=INIT,NET
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}" # set port based on SLURM_JOB_ID to avoid conflicts
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

CMD="export RANK=\$SLURM_PROCID && \
    export LOCAL_RANK=\$SLURM_LOCALID && \
    source venv/bin/activate && \
    python -m optimus.train \
    --model_name eurobert \
    --model_size base_nor_lumi \
    --tokenizer_path_or_name /scratch/project_465002893/nb-embed/EuroBERT/Australis \
    --data_mix_path data/mix \
    --output_dir ./runs --project_name australis \
    --lr 1e-4 \
    --num_epochs 1 \
    --batch_size 36 \
    --length 2048 \
    --mlm_probability 0.5 \
    --wandb_entity nbailab \
    --wandb_run_name test_ddp \
    --ddp True"

TDCMD="source venv/bin/activate &&
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 optimus/train.py \
    --model_name eurobert \
    --model_size base_nor_lumi \
    --tokenizer_path_or_name /scratch/project_465002893/nb-embed/EuroBERT/Australis \
    --data_mix_path data/mix \
    --output_dir ./runs --project_name australis \
    --lr 1e-4 \
    --num_epochs 1 \
    --batch_size 36 \
    --length 2048 \
    --mlm_probability 0.5 \
    --wandb_entity nbailab \
    --wandb_run_name test_ddp \
    --ddp True"

echo $CMD
# echo $TDCMD

# ls

srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity run $SIF bash -c "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && source venv2/bin/activate && python -m optimus.train --model_name eurobert --model_size base_nor_lumi --tokenizer_path_or_name /scratch/project_465002893/nb-embed/EuroBERT/Australis_128k --data_mix_path data/mix_128k --output_dir ./runs --project_name australis --lr 1e-3 --num_epochs 3 --batch_size 32 --length 2048 --mlm_probability 0.5 --wandb_entity nbailab --run_name 195m_128k_1e-3 --ddp True --fused_cross_entropy True --mixed_bfloat16 True --compile_model True --vocab_size 128000"

# srun singularity run $SIF bash -c $TDCMD
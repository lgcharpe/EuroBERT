#!/bin/bash
#SBATCH --account=project_465002893
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --time=2:00:00

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G

#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# set -euo pipefail

mkdir -p slurm_logs

module purge
module use /appl/local/laifs/modules 
module load lumi-aif-singularity-bindings
export HF_HOME=/scratch/project_465002893/nb-embed/huggingface_cache/
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif

singularity run $SIF bash -c "source venv/bin/activate && python -m optimus.tokenizer.train_from_old --old_tokenizer EuroBERT/EuroBERT-210m --new_tokenizer Australis_256k --dataset-path /scratch/project_465002893/nb-embed/aurora-2604-open-sample/data --dataset-type aurora --vocab-size 256000"
#!/bin/bash
#SBATCH --account=project_465002893
#SBATCH --partition=small
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=6G

#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# set -euo pipefail

mkdir -p slurm_logs

module purge
module use /appl/local/laifs/modules 
module load lumi-aif-singularity-bindings
export HF_HOME=/scratch/project_465002893/nb-embed/huggingface_cache/
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif


singularity run $SIF bash -c "source venv2/bin/activate && python -m optimus.dataprocess.tokenize_dataset --input_dir ../aurora-open/train --tokenizer Australis_128k --dataset aurora --output_dir full_data/train_tokenized_128k --num_workers 32 --size_limit 512MB"
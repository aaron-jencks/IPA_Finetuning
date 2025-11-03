#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=PAS2836
#SBATCH --output=/fs/ess/PAS2836/ipa_gpt/jobs/logs/%x-%j.out
#SBATCH --error=/fs/ess/PAS2836/ipa_gpt/jobs/logs/errors/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={cpus}
#SBATCH --time={timeout}
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus-per-node={gpus}

echo "===== [$(date)] JOB STARTED ====="

# Load required modules
module load miniconda3/24.1.2-py310 cuda/12.4.1
conda init bash
conda activate nanogpt_cu124  # TODO change this to your personal environment

echo "Python: $(which python) ($(python --version))"

# setup paths
scratch_prefix="/fs/scratch/PAS2836/ipa_gpt"
scratch_github_prefix="$scratch_prefix/github"
scratch_hf_cache_prefix="$scratch_prefix/cache"
mkdir -pv $scratch_github_prefix $scratch_hf_cache_prefix

repo_name="IPA_Finetuning"
repo_dir="$scratch_github_prefix/$repo_name"
cd "$repo_dir"

echo "===== [$(date)] RUNNING PYTHON SCRIPT ====="

# Run the actual script
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TQDM_DISABLE=1 python finetuning-exp-qa.py \
  "$SLURM_JOB_ID" --cpus {cpus} {args}

echo "===== [$(date)] CLEANING UP ====="

rm "$0" "{temp_config_name}"

echo "===== [$(date)] JOB COMPLETED ====="
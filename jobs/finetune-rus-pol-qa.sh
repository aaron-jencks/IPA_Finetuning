#!/bin/bash
#SBATCH --job-name=gpt2_finetune_rus_pol_qa
#SBATCH --account=PAS2836
#SBATCH --output=/fs/ess/PAS2836/ipa_gpt/jobs/logs/%x-%j.out
#SBATCH --error=/fs/ess/PAS2836/ipa_gpt/jobs/logs/errors/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus-per-node=1

echo "===== [$(date)] JOB STARTED ====="

# Load required modules
module load miniconda3/24.1.2-py310 cuda/12.4.1
conda init bash
conda activate nanogpt_cu124  # TODO change this to your personal environment

echo "Python: $(which python) ($(python --version))"

train_lang="both"
for arg in "$@"; do
  case $arg in
    --train-lang=*) train_lang="${arg#*=}";;
    *)
      echo "unknown argument: $arg"
      exit 1
      ;;
  esac
done

if [[ "$train_lang" == "both" ]]; then train_lang="russian polish"; fi
eval_lang="russian polish"

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
TQDM_DISABLE=1 python finetuning-exp-qa.py \
  "$SLURM_JOB_ID" config/finetune-rus-pol.json config/finetune-rus-pol-qa.json \
  --train-langs $train_lang --eval-langs $eval_lang --cpus 16

echo "===== [$(date)] JOB COMPLETED ====="

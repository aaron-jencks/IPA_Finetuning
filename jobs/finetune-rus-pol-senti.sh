#!/bin/bash
#SBATCH --job-name=gpt2_finetune_rus_pol_senti
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
eval_lang="both"
for arg in "$@"; do
  case $arg in
    --train-lang=*) train_lang="${arg#*=}";;
    --eval-lang=*) eval_lang="${arg#*=}";;
    *)
      echo "unknown argument: $arg"
      exit 1
      ;;
  esac
done

# setup paths
scratch_prefix="/fs/scratch/PAS2836/ipa_gpt"
scratch_github_prefix="$scratch_prefix/github"
scratch_hf_cache_prefix="$scratch_prefix/cache"
mkdir -pv $scratch_github_prefix $scratch_hf_cache_prefix

repo_name="IPA_Finetuning"
repo_address="git@github.com:aaron-jencks/$repo_name.git"
repo_branch="trainer"
repo_dir="$scratch_github_prefix/$repo_name"
if [ ! -d "$repo_dir" ]; then
  cd "$scratch_github_prefix"
  git clone "$repo_address"
  cd "$repo_name"
  git checkout "$repo_branch"
else
  cd "$repo_dir"
  git pull
fi

echo "===== [$(date)] RUNNING PYTHON SCRIPT ====="

# Run the actual script
TQDM_DISABLE=1 python finetuning-exp.py \
  "$SLURM_JOB_ID" "sentiment" \
  russian_polish_ipa_12_5_50k russian_polish_normal_12_5_50k \
  bpe-rus-pol-ipa-number-preservation bpe-rus-pol-normal-number-preservation \
  rus pol \
  iggy12345/brfrd-ipa iggy12345/allegro-reviews-ipa \
  --lang-1-features review \
  --lang-2-features text \
  --train-lang "$train_lang" \
  --eval-lang "$eval_lang" \
  --eval-feature five_class_label rating \
  --num-classes 5

echo "===== [$(date)] JOB COMPLETED ====="
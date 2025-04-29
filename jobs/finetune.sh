#!/bin/bash
#SBATCH --job-name=ipa_finetune_english
#SBATCH --account=PAS2836
#SBATCH --output=/fs/ess/PAS2836/ipa_gpt/jobs/logs/%x-%j.out
#SBATCH --error=/fs/ess/PAS2836/ipa_gpt/jobs/logs/errors/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus-per-node=1

echo "===== [$(date)] JOB STARTED ====="

# Load required modules
module load miniconda3/24.1.2-py310 cuda/12.4.1
conda init bash
conda activate nanogpt_cu124  # TODO change this to your personal environment

echo "Python: $(which python) ($(python --version))"

# setup paths
scratch_prefix="/fs/scratch/PAS2836/ipa_gpt"
storage_prefix="/fs/ess/PAS2836/ipa_gpt"
datasets_prefix="$storage_prefix/datasets"
checkpoints_prefix="$storage_prefix/checkpoints"
tokenizers_prefix="$storage_prefix/tokenizers"
scratch_datasets_prefix="$scratch_prefix/tokens"
scratch_github_prefix="$scratch_prefix/github"
mkdir -pv $scratch_datasets_prefix $scratch_github_prefix $checkpoints_prefix

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

task="sst2"
learning_rate="2e-5"
batch_size="8"
for arg in "$@"; do
  case $arg in
    --task=*) task="${arg#*=}";;
    --learning-rate=*) learning_rate="${arg#*=}";;
    --batch-size=*) batch_size="${arg#*=}";;
    *)
      echo "unknown argument: $arg"
      exit 1
      ;;
  esac
done

echo "Task: $task"
echo "Learning rate: $learning_rate"
echo "Batch size: $batch_size"

# Script specific names
model="openwebtext_normal_multi_node_12_5"
wandb_project="ipa-finetuning-english"
dataset="nyu-mll/glue"

checkpoint_path="$checkpoints_prefix/$model/ckpt.pt"
tokenizer_name="bpe-normal-number-preservation"
output_name="$task-lr$learning_rate-bs$batch_size"
output_path="$checkpoints_prefix/$output_name"

echo "===== [$(date)] RUNNING PYTHON SCRIPT ====="

# Run the actual script
python finetune.py \
  --vocab "$tokenizers_prefix/$tokenizer_name-vocab.json" \
  --merges "$tokenizers_prefix/$tokenizer_name-merges.txt" \
  --model "$checkpoint_path" \
  --task "$task" \
  --output "$output_path" \
  --learning-rate "$learning_rate" \
  --batch-size "$batch_size" \
  --hf-cache-dir "$datasets_prefix" \
  --dataset "$dataset" \
  --wandb-project "$wandb_project"

echo "===== [$(date)] JOB COMPLETED ====="
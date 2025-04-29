#!/bin/bash

# Usage: ./grid_submit.sh finetune.sh

# Grab the script filename from command-line arguments
script="$1"

# Check if the argument was given
if [ -z "$script" ]; then
    echo "Usage: $0 <script_to_run>"
    exit 1
fi

# Define grid
learning_rates=(1e-5 2e-5 3e-5 5e-5)
batch_sizes=(8 16 32 64 128)
tasks=("ax" "sst2" "mrpc" "rte" "qnli" "qqp" "cola" "wnli")

# Loop over all combinations
for task in "${tasks[@]}"; do
  for lr in "${learning_rates[@]}"; do
      for bs in "${batch_sizes[@]}"; do
          echo "Submitting job: $script --task $task --learning-rate $lr --batch-size $bs"
          sbatch "$script" "--task=$task" "--learning-rate=$lr" "--batch-size=$bs"
      done
  done
done

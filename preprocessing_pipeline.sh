#!/bin/bash

# Exit on error
set -e
set -x

# Step 1: Filter source and target files
echo "Running filtering..."
python3 MT-Preparation/filtering/filter.py data/en_files/source data/hi_files/target en hi

# Step 2: Train Unigram model on filtered files
echo "Training Unigram model..."
python3 MT-Preparation/subwording/1-train_unigram.py data/en_files/source-filtered.en data/hi_files/target-filtered.hi

# Step 3: Apply subword segmentation using the trained models
echo "Applying subword segmentation..."
python3 MT-Preparation/subwording/2-subword.py source.model target.model data/en_files/source-filtered.en data/hi_files/target-filtered.hi

# Step 4: Train-Test-Valid Split (Test and Dev sets have 2000 lines each)
echo "Spliting into Train, Test, Valid sets"
!python3 MT-Preparation/train_dev_split/train_dev_test_split.py 2000 2000 data/en_files/source-filtered.en.subword data/hi_files/target-filtered.hi.subword

echo "Pipeline completed successfully."

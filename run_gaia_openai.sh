#!/bin/bash

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh  # adjust this path if needed

cd ~/meta-researcher

# Activate your environment
conda activate meta-thinker

# Run your Python script, tee duplicates stdout to both terminal and log file
# stderr (2>) is redirected into stdout (1>&1) so it's also captured
python3 run_gaia_openai.py 2>&1 | tee run_gaia_openai.log

#!/bin/bash
# initialize conda environment 
source ~/miniconda3/etc/profile.d/conda.sh

conda activate neurosim
python relay_inference.py --model=$1
python run_neurosim.py --model=$1
python summary_neurosim.py --model=$1
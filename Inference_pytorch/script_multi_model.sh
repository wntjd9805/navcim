#!/bin/bash

# Check for correct number of arguments
if [ "$#" -ne 10 ]; then
    echo "Usage: $0 <models> <weights> <heterogeneity> <comb_heterogeneity> <topk> <step4_weights> <GA generation> <GA population> <search_accuracy> <guide_strategy>"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh

conda activate navcim

# Parse arguments
MODELS=$1
WEIGHTS=$2
HETEROGENEITY=$3
COMB_HETEROGENEITY=$4
TOPK=$5
WEIGHTS_STEP4=$6
GENERATION=$7
POPULATION=$8
SEARCH_ACCURACY=$9
STRATEGY=${10}


current_datetime=$(date +"%Yy%mM%dD_%Hh%Mm")

pueue group add process_kill
output=$(pueue add -g process_kill python $NAVCIM_DIR/process_kill.py)
pid=$(echo "$output" | grep -oP 'id \K\d+')

python create_log_folder.py --model=$MODELS --date=$current_datetime --search_accuracy=$SEARCH_ACCURACY --heterogeneity=$COMB_HETEROGENEITY --type="folder"
python create_log_folder.py --model=$MODELS --date=$current_datetime --search_accuracy=$SEARCH_ACCURACY --heterogeneity=$COMB_HETEROGENEITY --type="neurosim"
python create_log_folder.py --model=$MODELS --date=$current_datetime --search_accuracy=$SEARCH_ACCURACY --heterogeneity=$COMB_HETEROGENEITY --type="booksim"

# Convert comma-separated model names into array
IFS=',' read -r -a model_array <<< "$MODELS"

# Remove brackets and then split weights into an array of strings
clean_weights="${WEIGHTS//[\[\]]/}"
IFS=',' read -r -a weights_array <<< "$clean_weights"

# Step 1: Run profiling for each model with parsed weights
for i in "${!model_array[@]}"
do
    model="${model_array[$i]}"
    # Split individual model's weights into an array

    pueue group add $model
    pueue parallel -g $model 1
    pueue add --group $model "python profile_booksim_hetero_step1.py --model $model --latency ${weights_array[3*$i+0]} --power ${weights_array[3*$i+1]} --area ${weights_array[3*$i+2]} --heterogeneity $HETEROGENEITY --search_accuracy=$SEARCH_ACCURACY"
    pueue wait --group $model
done

# Step 2: Extract top-k configurations
python multi_model_extract_topk_step2.py --models $MODELS --heterogeneity $HETEROGENEITY --comb_heterogeneity $COMB_HETEROGENEITY --weights $WEIGHTS --topk $TOPK --search_accuracy=$SEARCH_ACCURACY

# Step 3: Use regression for each model with parsed weights
for i in "${!model_array[@]}"
do
    model="${model_array[$i]}"
    # Split individual model's weights into an array

    pueue add --group $model "python multi_model_use_regression_step3.py --models $MODELS --model $model --heterogeneity $COMB_HETEROGENEITY --latency ${weights_array[3*$i+0]} --power ${weights_array[3*$i+1]} --area ${weights_array[3*$i+2]} --search_accuracy=$SEARCH_ACCURACY"
    pueue wait --group $model
done

# Step 4: Perform search with genetic algorithm
clean_weights_step4="${WEIGHTS_STEP4//[\[\]]/}"
IFS=',' read -r -a weights_array_step4 <<< "$clean_weights_step4"

pueue group add $MODELS
pueue parallel -g $MODELS 1
pueue add --group $MODELS "python multi_model_search_GA_step4.py --models $MODELS --heterogeneity $COMB_HETEROGENEITY --weights $WEIGHTS --latency ${weights_array_step4[0]} --power ${weights_array_step4[1]} --area ${weights_array_step4[2]} --population_size $POPULATION --generation $GENERATION --search_accuracy=$SEARCH_ACCURACY --date=$current_datetime"
pueue wait --group $MODELS


if [ "$STRATEGY" = "constrain" ]; then
    echo "Constrain strategy"
    declare -a constrain_latency
    declare -a constrain_power
    declare -a constrain_area
    for i in "${!model_array[@]}"
    do
        model="${model_array[$i]}"
        pueue add --group $model "python profile_booksim_homo.py --model=$model --SA_size_1_ROW=128 --SA_size_1_COL=128 --PE_size_1=4 --TL_size_1=8"
        pueue wait --group $model

        FILE_PATH="//Inference_pytorch/search_result/${model}_homo/final_LATENCY_SA_row:128_SA_col:128_PE:4_TL:8.txt"

        IFS=',' read -r -a line_data < "$FILE_PATH"
    
        constrain_latency+=("${line_data[0]}")
        constrain_power+=("${line_data[1]}")
        constrain_area+=("${line_data[2]}")
    done
    constrain_latency=$(IFS=,; echo "${constrain_latency[*]}")
    constrain_power=$(IFS=,; echo "${constrain_power[*]}")
    constrain_area=$(IFS=,; echo "${constrain_area[*]}")
    python topsis_multimodel.py --model=$MODELS --heterogeneity=$HETEROGENEITY --latency=${weights_array_step4[0]} --power=${weights_array_step4[1]} --area=${weights_array_step4[2]} --constrain_latency=$constrain_latency --constrain_power=$constrain_power --constrain_area=$constrain_area --population_size $POPULATION --generation $GENERATION --search_accuracy=$SEARCH_ACCURACY --date=$current_datetime
else
    numbers=${STRATEGY#*[}  
    numbers=${numbers%]*}
    IFS=',' read -r -a weight <<< "$numbers"
  
    python topsis_multimodel.py --model=$MODELS --heterogeneity=$HETEROGENEITY --latency=${weights_array_step4[0]} --power=${weights_array_step4[1]} --area=${weights_array_step4[2]} --weight_latency=${weight[0]} --weight_power=${weight[1]} --weight_area=${weight[2]} --population_size $POPULATION --generation $GENERATION --search_accuracy=$SEARCH_ACCURACY --date=$current_datetime
fi

pueue kill $pid
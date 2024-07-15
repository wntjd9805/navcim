source ~/miniconda3/etc/profile.d/conda.sh

conda activate navcim
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <model> <latency> <power> <area> <heterogeneity> <search_accuracy> <guide_strategy>"
    exit 1
fi

MODEL_NAME=$1
LATENCY=$2
POWER=$3
AREA=$4
HETEROGENEITY=$5
SEARCH_ACCURACY=$6
STRATEGY=$7

current_datetime=$(date +"%Yy%mM%dD_%Hh%Mm")

pueue group add process_kill
output=$(pueue add -g process_kill python $NAVCIM_DIR/process_kill.py)
pid=$(echo "$output" | grep -oP 'id \K\d+')

python create_log_folder.py --model=$MODEL_NAME --date=$current_datetime --search_accuracy=$SEARCH_ACCURACY --heterogeneity=$HETEROGENEITY --type="folder"
python create_log_folder.py --model=$MODEL_NAME --date=$current_datetime --search_accuracy=$SEARCH_ACCURACY --heterogeneity=$HETEROGENEITY --type="neurosim"
python create_log_folder.py --model=$MODEL_NAME --date=$current_datetime --search_accuracy=$SEARCH_ACCURACY --heterogeneity=$HETEROGENEITY --type="booksim"

pueue group add $MODEL_NAME
pueue parallel -g $MODEL_NAME 1
pueue add --group $MODEL_NAME "python profile_booksim_hetero_step1.py  --model=$MODEL_NAME --latency=$LATENCY --power=$POWER --area=$AREA --heterogeneity=$HETEROGENEITY --search_accuracy=$SEARCH_ACCURACY"
pueue add --group $MODEL_NAME "python validate_booksim_hetero.py  --model=$MODEL_NAME --latency=$LATENCY --power=$POWER --area=$AREA --heterogeneity=$HETEROGENEITY --search_accuracy=$SEARCH_ACCURACY --date=$current_datetime" 

pueue wait --group $MODEL_NAME

if [ "$STRATEGY" = "constrain" ]; then
    pueue add --group $MODEL_NAME "python profile_booksim_homo.py --model=$MODEL_NAME --SA_size_1_ROW=128 --SA_size_1_COL=128 --PE_size_1=4 --TL_size_1=8"
    pueue wait --group $MODEL_NAME

    FILE_PATH="${NAVCIM_DIR}/Inference_pytorch/search_result/${MODEL_NAME}_homo/final_LATENCY_SA_row:128_SA_col:128_PE:4_TL:8.txt"

    IFS=',' read -r -a constrain < "$FILE_PATH"

    python topsis_singlemodel.py --model=$MODEL_NAME --heterogeneity=$HETEROGENEITY --latency=$LATENCY --power=$POWER --area=$AREA --search_accuracy=$SEARCH_ACCURACY --constrain_latency=${constrain[0]} --constrain_power=${constrain[1]} --constrain_area=${constrain[2]} --date=$current_datetime
else
    numbers=${STRATEGY#*[}  
    numbers=${numbers%]*}
    IFS=',' read -r -a weight <<< "$numbers"
  
    python topsis_singlemodel.py --model=$MODEL_NAME --heterogeneity=$HETEROGENEITY --latency=$LATENCY --power=$POWER --area=$AREA --search_accuracy=$SEARCH_ACCURACY --weight_latency=${weight[0]} --weight_power=${weight[1]} --weight_area=${weight[2]} --date=$current_datetime
fi

pueue kill $pid
#!/bin/bash

MODEL_NAME=$1

pueue group add $MODEL_NAME
pueue parallel -g $MODEL_NAME 10
pueue add --group $MODEL_NAME "python search_homogeneous_for_scale.py --model=$MODEL_NAME"

while [[ $(pueue status -g $MODEL_NAME | grep -E 'Queued|Running') ]]; do
    echo "Waiting for all jobs in group $MODEL_NAME to finish..."
    sleep 300  # 10초마다 확인
done

pueue parallel -g $MODEL_NAME 2
metrics=("energy" "latency")
trains=("mlp" "poly")
ADCs=(5)
Cellbits=(2)

for metric in ${metrics[@]}; do
  for train in ${trains[@]}; do
    for ADC in ${ADCs[@]}; do
      for Cellbit in ${Cellbits[@]}; do
        if [ "$metric" == "energy" ] && [ "$train" == "mlp" ]; then
          pueue add --group $MODEL_NAME "python booksim_regression_scalefactor.py --model=$MODEL_NAME --metric=$metric --train=$train --ADC=$ADC --Cellbit=$Cellbit"
        elif [ "$metric" == "latency" ] && [ "$train" == "poly" ]; then
          pueue add --group $MODEL_NAME "python booksim_regression_scalefactor.py --model=$MODEL_NAME --metric=$metric --train=$train --ADC=$ADC --Cellbit=$Cellbit"
        fi
      done
    done
  done
done

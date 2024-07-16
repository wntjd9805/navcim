import argparse
import csv
import os
import numpy as np
from sys import path
navcim_dir = os.getenv('NAVCIM_DIR')
path.append(f"{navcim_dir}/TOPSIS-Python/")
from topsis import Topsis
import math
import pandas as pd
import ast
from tabulate import tabulate
from datetime import datetime

a=math.inf
parser = argparse.ArgumentParser()
parser.add_argument("--model",  required=True)
parser.add_argument('--heterogeneity',type=int, required=True)
parser.add_argument('--distribute',type=int ,default=1, help='distribute')
parser.add_argument('--beam_size_m',type=int ,default=700,help='beam_size_m')
parser.add_argument('--beam_size_n',type=int ,default=3,help='beam_size_n')
parser.add_argument('--latency',type=int ,required=True)
parser.add_argument('--power',type=int ,required=True)
parser.add_argument('--area',type=int ,required=True)
parser.add_argument('--accuracy',type=int ,default=1,help='weight_accuracy_with_pareto')
parser.add_argument('--weight_latency',type=int ,default=1)
parser.add_argument('--weight_power',type=int ,default=1)
parser.add_argument('--weight_area',type=int ,default=1)
parser.add_argument('--weight_accuracy',type=int ,default=1,help='weight_accuracy')
parser.add_argument('--constrain_latency',type=float ,default=float('inf'))
parser.add_argument('--constrain_power',type=float ,default=float('inf'))
parser.add_argument('--constrain_area',type=float ,default=float('inf'))
parser.add_argument('--search_accuracy',type=int ,default=0, help='search_accuracy')
parser.add_argument('--search_accuracy_metric', type=str, default='cka', choices=['mse', 'cosine', 'ssim', 'cka'], help='metric')
parser.add_argument('--display',type=int ,default=20, help='How much to show')
parser.add_argument('--date', default="default")
parser.add_argument('--start_time', default=None)


args = parser.parse_args()

line_length = 60

CONFIG_pareto = []
paretoPoints_list=[]

if args.search_accuracy == 0:
    fname = f"{navcim_dir}/Inference_pytorch/search_result/{args.model}_hetero_validate/final_result_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area}]_{args.heterogeneity}.txt"
    with open(fname) as f:
        lines = f.readlines()
    increase = 0.01
    cnt = 0
    while not (paretoPoints_list and CONFIG_pareto):
      for l in lines:
          if float(l.split("\n")[0].split("\"")[2].split(",")[1:][0])<args.constrain_latency + args.constrain_latency*increase*cnt: #latency
              if float(l.split("\n")[0].split("\"")[2].split(",")[1:][1]) <args.constrain_power + args.constrain_power*increase*cnt:  #power
                  if float(l.split("\n")[0].split("\"")[2].split(",")[1:][2]) < args.constrain_area + args.constrain_area*increase*cnt: #area
                      paretoPoints_list.append(l.split("\n")[0].split("\"")[2].split(",")[1:][:3])
                      CONFIG_pareto.append(l.split("\n")[0].split("\"")[1])
      cnt += 1
    
    w = [args.weight_latency,args.weight_power,args.weight_area]
    if len(paretoPoints_list) == 0:
        print("There is no data that satisfies the constraints.")
        quit()
    sign = np.array([False,False,False])
    t = Topsis(paretoPoints_list, w, sign)
    t.calc()

else:
    fname = f"{navcim_dir}/Inference_pytorch/search_result/{args.model}_hetero_validate/final_result_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area},{args.accuracy}]_{args.heterogeneity}_{args.search_accuracy_metric}.txt"
    with open(fname) as f:
        lines = f.readlines()
    
    increase = 0.01
    cnt = 0
    while not (paretoPoints_list and CONFIG_pareto):
      for l in lines:
          if cnt == 0:
             if float(l.split("\n")[0].split("\"")[2].split(",")[1:][0])<args.constrain_latency: #latency
                if float(l.split("\n")[0].split("\"")[2].split(",")[1:][1]) <args.constrain_power:  #power
                    if float(l.split("\n")[0].split("\"")[2].split(",")[1:][2]) < args.constrain_area : #area
                        paretoPoints_list.append(l.split("\n")[0].split("\"")[2].split(",")[1:])
                        CONFIG_pareto.append(l.split("\n")[0].split("\"")[1])
          else:
            if float(l.split("\n")[0].split("\"")[2].split(",")[1:][0])<args.constrain_latency + args.constrain_latency*increase*cnt: #latency
                if float(l.split("\n")[0].split("\"")[2].split(",")[1:][1]) <args.constrain_power + args.constrain_power*increase*cnt:  #power
                    if float(l.split("\n")[0].split("\"")[2].split(",")[1:][2]) < args.constrain_area + args.constrain_area*increase*cnt: #area
                        paretoPoints_list.append(l.split("\n")[0].split("\"")[2].split(",")[1:])
                        CONFIG_pareto.append(l.split("\n")[0].split("\"")[1])
      cnt += 1

    w = [args.weight_latency,args.weight_power,args.weight_area,args.accuracy]
    sign = np.array([False,False,False,True])
    t = Topsis(paretoPoints_list, w, sign)
    t.calc()



if args.search_accuracy == 0:
  columns = ['Tile1', 'Tile2', 'ADC Precision', 'Cellbit', 'Latency (ns)', 'Power (mW)', 'Area (um^2)']
else:
  columns = ['Tile1', 'Tile2', 'ADC Precision', 'Cellbit', 'Latency (ns)', 'Power (mW)', 'Area (um^2)', 'Accuracy (%)']

if args.constrain_latency != a and args.constrain_power != a and args.constrain_area != a:
   args.display = len(t.rank_to_best_similarity())

sa_set = []
pe_set = 0
tile_set = 0
adc_set = []
cellbit_set = []
with open(f"./search_space.txt") as f:
    lines = f.readlines()
    a = lines[0].split("=")[1].strip().split(',')
    for i in a:
        sa_set.append(int(i))
    pe_set = int(lines[1].split("=")[1].strip())
    tile_set = int(lines[2].split("=")[1].strip())
    a = lines[3].split("=")[1].strip().split(',')
    for i in a:
        adc_set.append(int(i))
    a = lines[4].split("=")[1].strip().split(',')
    for i in a:
        cellbit_set.append(int(i))


if args.search_accuracy == 0:
        output_file = f"NavCim_log/{args.model}/accuracy_false/ADC_{adc_set}/CellBit_{cellbit_set}/Tile_{tile_set}/PE_{pe_set}/SA_{sa_set}/heterogeneity_{args.heterogeneity}/{args.date}/Navcim_search_result.txt"
else:   
    output_file = f"NavCim_log/{args.model}/accuracy_true/Tile_{tile_set}/PE_{pe_set}/SA_{sa_set}/ADC_{adc_set}/CellBit_{cellbit_set}/heterogeneity_{args.heterogeneity}/{args.date}/Navcim_search_result.txt"


with open(output_file, 'w') as log_file:
  if cnt > 1 :
    log_file.write("=" * line_length+"\n")
    log_file.write("There is no data that satisfies the constraints. So we will find the data that satisfies the constraints with a little more relaxed constraints.\n")
  if len(t.rank_to_best_similarity()) < args.display:
    args.display = len(t.rank_to_best_similarity())
  # Display Setup Information and Parameters
  log_file.write(" Setup Information and Parameters ".center(line_length, "=")+"\n")
  log_file.write(f"Model          : {args.model}\n")
  log_file.write(f"Weight         : Latency = {args.weight_latency}, Power = {args.weight_power}, Area = {args.weight_area}\n")
  log_file.write(f"Heterogeneity  : {args.heterogeneity}\n")
  if args.search_accuracy == 1:
      log_file.write(f"Accuracy       : True\n")
  else:
      log_file.write(f"Accuracy       : False\n")

  if cnt > 1:
      log_file.write(f"Relaxation     : {increase*cnt*100}%\n")
  log_file.write(f"Constraint     : Latency < {args.constrain_latency}, Power < {args.constrain_power}, Area < {args.constrain_area}\n")
  # Display Search Space from the search_space.txt
  sa_set = []
  pe_set = 0
  tile_set = 0
  adc_set = []
  cellbit_set = []

  # Read and parse the search space parameters from file
  if os.path.exists("./search_space.txt"):
      with open("./search_space.txt") as f:
          sa_set = f.readline().strip().split('=')[1].strip()
          pe_set = int(f.readline().strip().split('=')[1].strip())
          tile_set = int(f.readline().strip().split('=')[1].strip())
          adc_bit = int(f.readline().strip().split('=')[1].strip())
          cell_bit = int(f.readline().strip().split('=')[1].strip())

      # Calculate PE and Tile ranges based on set values
      pe_range = f"2 ≤ n ≤ {pe_set}"
      tile_range = f"2 ≤ m ≤ {tile_set}"

      # Display Search Space
      log_file.write(" Search Space ".center(line_length, "=")+"\n")
      log_file.write(f"SA Width       : {sa_set}\n")
      log_file.write(f"SA Height      : {sa_set}\n")
      log_file.write(f"PE Size        : SA Width × n ({pe_range}) ∀n (m % n = 0)\n")
      log_file.write(f"Tile Size      : SA Width × m ({tile_range})\n")
      log_file.write(f"ADC Precision  : {adc_bit}\n")
      log_file.write(f"Cellbit        : {cell_bit}\n")


  log_file.write(" Search Result Summary ".center(line_length, "=")+"\n")
  if args.start_time == None:
    start_time = datetime.now()
  else:
    start_time = datetime.strptime(args.start_time, "%a %b %d %H:%M:%S %Z %Y")

  log_file.write(f"Exploration started at {start_time}\n")
  end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  log_file.write(f"Exploration ended at {end_time}\n")
  execution_time = datetime.now() - start_time
  log_file.write(f"Exploration time: {execution_time}\n")

  df = pd.DataFrame(columns=columns)
  for idx in range(args.display):
      config_str, sequence_str = CONFIG_pareto[t.rank_to_best_similarity()[idx]-1].split('_')[0], CONFIG_pareto[t.rank_to_best_similarity()[idx]-1].split('_')[1:]
      sequence_str[-1] = sequence_str[-1].split(',')[0]
      config_list = ast.literal_eval(config_str)
    
      if args.search_accuracy == 0:
        latency = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][0])
        power = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][1])
        area = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][2])
        tile1_details = f"SA_width={config_list[0][0]}, SA_height={config_list[0][1]}, PE_Size={config_list[0][0] * config_list[0][2]}, Tile_Size={config_list[0][0] * config_list[0][3]}"
        tile2_details = f"SA_width={config_list[1][0]}, SA_height={config_list[1][1]}, PE_Size={config_list[1][0] * config_list[1][2]}, Tile_Size={config_list[1][0] * config_list[1][3]}"
        adc_precision = f"{config_list[0][4]}"
        cellbit = f"{config_list[0][5]}"

        new_row = pd.DataFrame({
            'Tile1': [tile1_details],
            'Tile2': [tile2_details],
            'ADC Precision': [adc_precision],
            'Cellbit': [cellbit],
            'Latency (ns)': [latency],
            'Power (mW)': [power],
            'Area (um^2)': [area]
        }, columns=columns)
      else:
        latency = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][0])
        power = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][1])
        area = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][2])
        accuracy = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][3])
        tile1_details = f"SA_width={config_list[0][0]}, SA_height={config_list[0][1]}, PE_Size={config_list[0][0] * config_list[0][2]}, Tile_Size={config_list[0][0] * config_list[0][3]}"
        tile2_details = f"SA_width={config_list[1][0]}, SA_height={config_list[1][1]}, PE_Size={config_list[1][0] * config_list[1][2]}, Tile_Size={config_list[1][0] * config_list[1][3]}"
        adc_precision = f"{config_list[0][4]}"
        cellbit = f"{config_list[0][5]}"
        
        new_row = pd.DataFrame({
            'Tile1': [tile1_details],
            'Tile2': [tile2_details],
            'ADC Precision': [adc_precision],
            'Cellbit': [cellbit],
            'Latency (ns)': [latency],
            'Power (mW)': [power],
            'Area (um^2)': [area],
            'Accuracy (%)': [accuracy]
        }, columns=columns)
      
      
      df = pd.concat([df, new_row], ignore_index=True)

      # log_file.write each log's detail in a formatted way
      log_file.write(f"TOP {idx+1} Data Log:\n")
      log_file.write(tabulate(df.iloc[[idx]], headers='keys', tablefmt='grid', showindex=False)+"\n")
      sequence = ', '.join([f"Layer {i+1}: Tile{int(s)+1}" for i, s in enumerate(sequence_str)])
      log_file.write(f"Data Log {idx+1} Layer mapping result:\n")
      log_file.write(sequence+"\n")
      log_file.write("\n")

print(f"please check the log and search result file in {output_file}")
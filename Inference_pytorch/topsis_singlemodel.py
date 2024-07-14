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


args = parser.parse_args()



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

# print("best_similarity\t", t.best_similarity[t.rank_to_best_similarity()[0]-1])

if args.search_accuracy == 0:
  columns = ['Tile1', 'Tile2', 'ADC Precision', 'Cellbit', 'Latency (ns)', 'Power (mW)', 'Area (um^2)']
else:
  columns = ['Tile1', 'Tile2', 'ADC Precision', 'Cellbit', 'Latency (ns)', 'Power (mW)', 'Area (um^2)', 'Accuracy (%)']

if args.constrain_latency != a and args.constrain_power != a and args.constrain_area != a:
   args.display = len(t.rank_to_best_similarity())

if cnt > 1 :
  print("=====================================================")
  print("There is no data that satisfies the constraints. So we will find the data that satisfies the constraints with a little more relaxed constraints.")
if len(t.rank_to_best_similarity()) < args.display:
  args.display = len(t.rank_to_best_similarity())
# Display Setup Information and Parameters
print("================= Setup Information and Parameters ==================")
print(f"Model          : {args.model}")
print(f"Weight         : Latency = {args.weight_latency}, Power = {args.weight_power}, Area = {args.weight_area}")
print(f"Heterogeneity  : {args.heterogeneity}")
if args.search_accuracy == 1:
    print(f"Accuracy       : True")
else:
    print(f"Accuracy       : False")

if cnt > 1:
    print(f"Relaxation     : {increase*cnt*100}%")
print(f"Constraint     : Latency < {args.constrain_latency}, Power < {args.constrain_power}, Area < {args.constrain_area}")
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
    print("================= Search Space ==================")
    print(f"SA Width       : {sa_set}")
    print(f"SA Height      : {sa_set}")
    print(f"PE Size        : SA Width × n ({pe_range}) ∀n (m % n = 0)")
    print(f"Tile Size      : SA Width × m ({tile_range})")
    print(f"ADC Precision  : {adc_bit}")
    print(f"Cellbit        : {cell_bit}")


print("================= Search Result Summary ==================")
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

    # Print each log's detail in a formatted way
    print(f"TOP {idx+1} Data Log:")
    print(tabulate(df.iloc[[idx]], headers='keys', tablefmt='grid', showindex=False))
    sequence = ', '.join([f"Layer {i+1}: Tile{int(s)+1}" for i, s in enumerate(sequence_str)])
    print(f"Data Log {idx+1} Layer mapping result:")
    print(sequence)
    print()
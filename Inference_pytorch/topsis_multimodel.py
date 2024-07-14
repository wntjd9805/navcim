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
parser.add_argument("--models",  required=True)
parser.add_argument('--heterogeneity',type=int, required=True)
parser.add_argument('--distribute',type=int ,default=1, help='distribute')
parser.add_argument('--beam_size_m',type=int ,default=700,help='beam_size_m')
parser.add_argument('--beam_size_n',type=int ,default=3,help='beam_size_n')
parser.add_argument('--latency',type=int ,required=True)
parser.add_argument('--power',type=int ,required=True)
parser.add_argument('--area',type=int ,required=True)
parser.add_argument('--accuracy',type=int )
parser.add_argument('--weight_latency',type=int ,default=1)
parser.add_argument('--weight_power',type=int ,default=1)
parser.add_argument('--weight_area',type=int ,default=1)
parser.add_argument('--constrain_latency',type=str ,default=None)
parser.add_argument('--constrain_power',type=str ,default=None)
parser.add_argument('--constrain_area',type=str ,default=None)
parser.add_argument('--search_accuracy',type=int ,default=0, help='search_accuracy')
parser.add_argument('--search_accuracy_metric', type=str, default='cka', choices=['mse', 'cosine', 'ssim', 'cka'], help='metric')
parser.add_argument('--population_size',type=int ,default=5, help='Population size in a generation')
parser.add_argument('--generation',type=int ,default=3, help='Total number of generations')
parser.add_argument('--display',type=int ,default=20, help='How much to show')

args = parser.parse_args()
   
CONFIG_dic = {}
paretoPoints_dic={}
model_list = args.models.split(',')


constrain_latency = [float('inf') for model in model_list]
constrain_power = [float('inf') for model in model_list]
constrain_area = [float('inf') for model in model_list]
if args.constrain_latency != None:
    constrain_latency = [float(item) for item in args.constrain_latency.split(',')]
if args.constrain_power != None:
    constrain_power = [float(item) for item in args.constrain_power.split(',')]
if args.constrain_area != None:
    constrain_area = [float(item) for item in args.constrain_area.split(',')]


if args.search_accuracy == 0:
    for model in model_list:
      CONFIG =[]
      paretoPoints=[]
      fname = f"{navcim_dir}/Inference_pytorch/search_result/{args.models}_model_search_result_GA_predict/final_result_{model}_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area}]_{args.heterogeneity}_{args.generation}_{args.population_size}_GA.txt"
      with open(fname) as f:
        lines = f.readlines()
      for l in lines:
          CONFIG.append(l.split("\n")[0].split('","')[0][1:]+','+l.split("\n")[0].split('","')[1])    
          paretoPoints.append(ast.literal_eval(l.split("\n")[0].split('","')[2][:-1]))
      CONFIG_dic[model]=CONFIG
      paretoPoints_dic[model]=paretoPoints
else:
    for model in model_list:
      # print(model)
      CONFIG =[]
      paretoPoints=[]
      fname = f"{navcim_dir}/Inference_pytorch/search_result/{args.models}_model_search_result_GA_predict/final_result_{model}_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area},{args.accuracy}]_{args.heterogeneity}_{args.generation}_{args.population_size}_GA.txt"
      with open(fname) as f:
        lines = f.readlines()
      for l in lines:
          CONFIG.append(l.split("\n")[0].split('","')[0][1:]+','+l.split("\n")[0].split('","')[1])    
          paretoPoints.append(ast.literal_eval(l.split("\n")[0].split('","')[2][:-1]))
      CONFIG_dic[model]=CONFIG
      paretoPoints_dic[model]=paretoPoints

CONFIG_pareto = []
paretoPoints_list=[]
increase = 0.01
cnt = 0
while not (paretoPoints_list and CONFIG_pareto):
    for idc in range(len(paretoPoints_dic[model_list[0]])):
        all_models_satisfy = True  
        for i,model in enumerate(model_list):
            latency = float(paretoPoints_dic[model][idc][0])
            power = float(paretoPoints_dic[model][idc][1])
            area = float(paretoPoints_dic[model][idc][2])
            if cnt == 0:
              if not (latency < constrain_latency[i] and
                      power < constrain_power[i] and
                      area < constrain_area[i] ):
                  all_models_satisfy = False
                  break  # 하나의 모델이라도 조건을 만족하지 않으면 루프를 빠져나옴
            else:
              if not (latency < constrain_latency[i] + constrain_latency[i] * increase * cnt and
                      power < constrain_power[i] + constrain_power[i] * increase * cnt and
                      area < constrain_area[i] + constrain_area[i] * increase * cnt):
                  all_models_satisfy = False
                  break  # 하나의 모델이라도 조건을 만족하지 않으면 루프를 빠져나옴

        if all_models_satisfy:
            point_tmp=[]
            config_tmp=[]
            for model in model_list:
               point_tmp.append(paretoPoints_dic[model][idc])
               config_tmp.append(CONFIG_dic[model][idc])
            
            paretoPoints_list.append(point_tmp)
            CONFIG_pareto.append(config_tmp)

    cnt += 1


topsis_point=[]

if args.search_accuracy == 0:
    for points in paretoPoints_list:
      latency = 0
      power = 0
      area = 0
      for point in points:
         latency += float(point[0])
         power += float(point[1])
         area = float(point[2]) if area<float(point[2]) else area
      topsis_point.append([latency,power,area])
else:
    for points in paretoPoints_list:
      latency = 0
      power = 0
      area = 0
      accuracy = 0
      # print(points)
      for point in points:
         latency += float(point[0])
         power += float(point[1])
         area = float(point[2]) if area<float(point[2]) else area
         accuracy += float(point[4])
      topsis_point.append([latency,power,area,accuracy])

if args.search_accuracy == 0:
  w = [args.weight_latency,args.weight_power,args.weight_area]
  sign = np.array([False,False,False])
  t = Topsis(topsis_point, w, sign)
  t.calc()
  columns = ['Model', 'Tile1', 'Tile2', 'ADC Precision', 'Cellbit', 'Latency (ns)', 'Power (mW)', 'Area (um^2)']

else:
  w = [args.weight_latency,args.weight_power,args.weight_area,args.accuracy]
  sign = np.array([False,False,False,True])
  t = Topsis(topsis_point, w, sign)
  t.calc()
  columns = ['Model', 'Tile1', 'Tile2', 'ADC Precision', 'Cellbit', 'Latency (ns)', 'Power (mW)', 'Area (um^2)', 'Accuracy (%)']


if cnt > 1:
  print("=====================================================")
  print("There is no data that satisfies the constraints. So we will find the data that satisfies the constraints with a little more relaxed constraints.")
if len(t.rank_to_best_similarity()) < args.display:
  args.display = len(t.rank_to_best_similarity())
# Display Setup Information and Parameters
print("================= Setup Information and Parameters ==================")
print(f"Model          : {args.models}")
print(f"Weight         : Latency = {args.weight_latency}, Power = {args.weight_power}, Area = {args.weight_area}")
print(f"Heterogeneity  : {args.heterogeneity}")
print(f"GA Generation  : {args.generation}")
print(f"GA Population  : {args.population_size}")
if args.search_accuracy == 1:
    print(f"Accuracy       : True")
else:
    print(f"Accuracy       : False")

if cnt > 1:
    print(f"Relaxation     : {increase*cnt*100}%")
print(f"Constraint     : Latency < {constrain_latency}, Power < {constrain_power}, Area < {constrain_area}")
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
row = 0
for idx in range(args.display):
    area = 0
    sequence_list = {}
    for j,model in enumerate(model_list):
        config_str, sequence_str = CONFIG_pareto[t.rank_to_best_similarity()[idx]-1][j].split(']],')[0]+']]', CONFIG_pareto[t.rank_to_best_similarity()[idx]-1][j].split(']],')[1]
        config_list = ast.literal_eval(config_str)
        sequence_list[model]= ast.literal_eval(sequence_str)
        
        if args.search_accuracy == 0:
          latency = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][j][0])
          power = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][j][1])
          area = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][j][2]) if area < float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][j][2]) else area
          tile1_details = f"SA_width={config_list[0][0]}, SA_height={config_list[0][1]}, PE_Size={config_list[0][0] * config_list[0][2]}, Tile_Size={config_list[0][0] * config_list[0][3]}"
          tile2_details = f"SA_width={config_list[1][0]}, SA_height={config_list[1][1]}, PE_Size={config_list[1][0] * config_list[1][2]}, Tile_Size={config_list[1][0] * config_list[1][3]}"
          adc_precision = f"{config_list[0][4]}"
          cellbit = f"{config_list[0][5]}"
          
          new_row = pd.DataFrame({
              'Model': [model],
              'Tile1': [tile1_details],
              'Tile2': [tile2_details],
              'ADC Precision': [adc_precision],
              'Cellbit': [cellbit],
              'Latency (ns)': [latency],
              'Power (mW)': [power],
              'Area (um^2)': [area]
          }, columns=columns)
        else:
          latency = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][j][0])
          power = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][j][1])
          area = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][j][2]) if area < float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][j][2]) else area
          accuracy = float(paretoPoints_list[t.rank_to_best_similarity()[idx]-1][j][4])
          tile1_details = f"SA_width={config_list[0][0]}, SA_height={config_list[0][1]}, PE_Size={config_list[0][0] * config_list[0][2]}, Tile_Size={config_list[0][0] * config_list[0][3]}"
          tile2_details = f"SA_width={config_list[1][0]}, SA_height={config_list[1][1]}, PE_Size={config_list[1][0] * config_list[1][2]}, Tile_Size={config_list[1][0] * config_list[1][3]}"
          adc_precision = f"{config_list[0][4]}"
          cellbit = f"{config_list[0][5]}"
          
          new_row = pd.DataFrame({
              'Model': [model],
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
    print(tabulate(df.iloc[row:row+2], headers='keys', tablefmt='grid',  showindex=False))
    
    for model in model_list:
      print(f"{model} Layer mapping result:")
      sequence = ', '.join([f"Layer {i+1}: Tile{int(s)+1}" for i, s in enumerate(sequence_list[model])])
      print(sequence)
      print()
    print('\n\n')
    row += 2
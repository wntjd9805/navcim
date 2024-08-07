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
parser.add_argument("--models",  required=True)
parser.add_argument('--heterogeneity',type=int, required=True)
parser.add_argument('--distribute',type=int ,default=1, help='distribute')
parser.add_argument('--beam_size_m',type=int ,default=700,help='beam_size_m')
parser.add_argument('--beam_size_n',type=int ,default=3,help='beam_size_n')
parser.add_argument('--latency',type=int ,required=True)
parser.add_argument('--power',type=int ,required=True)
parser.add_argument('--area',type=int ,required=True)
parser.add_argument('--accuracy',type=int ,default=1)
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
parser.add_argument('--date', default="default")
parser.add_argument('--start_time', default=None)
args = parser.parse_args()

line_length = 60

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
        output_file = f"NavCim_log/{args.models}/accuracy_false/ADC_{adc_set}/CellBit_{cellbit_set}/Tile_{tile_set}/PE_{pe_set}/SA_{sa_set}/heterogeneity_{args.heterogeneity}/{args.date}/Navcim_search_result.txt"
else:   
    output_file = f"NavCim_log/{args.models}/accuracy_true/Tile_{tile_set}/PE_{pe_set}/SA_{sa_set}/ADC_{adc_set}/CellBit_{cellbit_set}/heterogeneity_{args.heterogeneity}/{args.date}/Navcim_search_result.txt"


with open(output_file, 'w') as log_file:
  if len(t.rank_to_best_similarity()) < args.display:
    args.display = len(t.rank_to_best_similarity())
  # Display Setup Information and Parameters
  log_file.write(" Setup Information and Parameters ".center(line_length, "=")+"\n")
  log_file.write(f"Input models       : {args.models}\n")
  log_file.write(f"Weights            : Latency = {args.weight_latency}, Power = {args.weight_power}, Area = {args.weight_area}\n")
  log_file.write(f"Tile heterogeneity : {args.heterogeneity}\n")
  log_file.write(f"GA Generation      : {args.generation}\n")
  log_file.write(f"GA Population      : {args.population_size}\n")
  if args.search_accuracy == 1:
      log_file.write(f"Accuracy           : True\n")
  else:
      log_file.write(f"Accuracy           : False\n")

  log_file.write(f"Search constraints (user input) : Latency < {constrain_latency}, Power < {constrain_power}, Area < {constrain_area}\n")
  if cnt > 1:
      relaxed_constrain_latency = [constrain_latency[i] + constrain_latency[i] * increase * (cnt-1) for i in range(len(constrain_latency))]
      relaxed_constrain_power = [constrain_power[i] + constrain_power[i] * increase * (cnt-1) for i in range(len(constrain_power))]
      relaxed_constrain_area = [constrain_area[i] + constrain_area[i] * increase * (cnt-1) for i in range(len(constrain_area))]
      log_file.write("There is no data that satisfies the constraints. So we will find the data that satisfies the constraints with a little more relaxed constraints.\n")
      log_file.write(f"Search constraints (applied)    : Latency < {relaxed_constrain_latency}, Power < {relaxed_constrain_power}, Area < {relaxed_constrain_area}\n")
      log_file.write(f"Relaxation     : {increase*(cnt-1)*100}%\n")
  
  # Display Search Space from the search_space.txt

  pe_range = f"2 ≤ n ≤ {pe_set}"
  tile_range = f"2 ≤ m ≤ {tile_set}"

  # Display Search Space
  log_file.write(" Search Space ".center(line_length, "=")+'\n')
  log_file.write(f"SA Width             : {sa_set}\n")
  log_file.write(f"SA Height            : {sa_set}\n")
  log_file.write(f"PE Size              : SA Width × n ({pe_range}) ∀n (m % n = 0)\n")
  log_file.write(f"Tile Size            : SA Width × m ({tile_range})\n")
  log_file.write(f"ADC Precision (bits) : {adc_set}\n")
  log_file.write(f"Cellbit              : {cellbit_set}\n")


  log_file.write(" Search Result Summary ".center(line_length, "=")+'\n')

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
      log_file.write(f"TOP {idx+1} Data Log:\n")
      log_file.write(tabulate(df.iloc[row:row+2], headers='keys', tablefmt='grid',  showindex=False)+'\n')
      
      for model in model_list:
        log_file.write(f"{model} Layer mapping result:\n")
        sequence = ', '.join([f"Layer {i+1}: Tile{int(s)+1}" for i, s in enumerate(sequence_list[model])])
        log_file.write(sequence+'\n')
        log_file.write('\n')
      log_file.write('\n\n')
      row += 2
print(f"please check the log and search result file in {output_file}")
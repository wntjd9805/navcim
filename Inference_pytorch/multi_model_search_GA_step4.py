import argparse
from audioop import ratecv
import os
from sqlite3 import Row
import time
from utee import misc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from utee import wage_util
# from models import dataset
import torchvision.models as models
import shutil
import sys
import multiprocessing
#from IPython import embed
from datetime import datetime
from subprocess import call

import tvm 
import tvm.relay as relay
import onnx
from parse import *
import re
import csv
import math
from itertools import combinations
import numpy as np
import ctypes
import copy
import subprocess
import random
import math
from multiprocessing import Pool,Manager, Value, Lock
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import ast
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
import torch.nn as nn
import joblib
from sklearn.preprocessing import PolynomialFeatures
import pickle
from sys import path

navcim_dir = os.getenv('NAVCIM_DIR')
path.append(f"{navcim_dir}/TOPSIS-Python/")
from topsis import Topsis

import traceback
path.append(f'{navcim_dir}/cross-sim/applications/dnn/inference')
import run_inference_for_search
import pandas as pd
from copy import deepcopy

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--models', default='ResNet50', help='e.g. VGG16,ResNet50')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 64)')
parser.add_argument('--heterogeneity', type=int, default=2, help='heterogeneity')
parser.add_argument('--distribute',type=int ,default=1, help='distribute')
parser.add_argument('--beam_size_m',type=int ,default=700,help='beam_size_m')
parser.add_argument('--beam_size_n',type=int ,default=3,help='beam_size_n')
parser.add_argument('--weights',type=str ,required=True,help='consist of [latency,power,area] with pareto weight')
parser.add_argument('--latency',type=int ,default=1,help='weight_latency_with_pareto')
parser.add_argument('--power',type=int ,default=1,help='weight_power_with_pareto')
parser.add_argument('--area',type=int ,default=1,help='weight_area_with_pareto')
parser.add_argument('--accuracy',type=int ,default=1,help='weight_acc_with_pareto')
parser.add_argument('--search_accuracy',type=int ,default=0, help='search_accuracy')
parser.add_argument('--population_size',type=int ,default=10, help='Population size in a generation')
parser.add_argument('--mutation_rate',type=float ,default=0.1, help='rate of mutation')
parser.add_argument('--generation',type=int ,default=3, help='Total number of generations')
parser.add_argument('--date', default="default")
args = parser.parse_args()
random.seed(42)

def print_configuration_message(config, file_path):
    with open(file_path, 'w') as file:
        file.write("=======================================\n")
        file.write(config.task + " sim: " + str(config.ntest) + " images, start: " + str(config.nstart) + "\n")
        file.write("Mapping: " + str(config.style) + "\n")
        if config.style == "BALANCED":
            if config.Nslices == 1:
                file.write('  Differential cells style: ' + config.balanced_style + "\n")
            if config.interleaved_posneg:
                file.write('  Positive/negative weight cells interleaved on column\n')
            else:
                file.write('  Subtract current in crossbar: ' + str(config.subtract_current_in_xbar) + "\n")
        if config.style == "OFFSET":
            file.write('  Digital offset: ' + str(config.digital_offset) + "\n")
        if config.weight_bits > 0:
            file.write('  Weight quantization: ' + str(config.weight_bits) + ' bits\n')
        else:
            file.write('  Weight quantization off\n')
        
        file.write('  Digital bias: ' + str(config.digital_bias) + "\n")
        file.write('  Batchnorm fold: ' + str(config.fold_batchnorm) + "\n")
        if config.bias_bits == "adc":
            file.write('  Bias quantization: following ADC\n')
        elif config.bias_bits > 0:
            file.write('  Bias quantization: ' + str(config.bias_bits) + ' bits\n')
        else:
            file.write('  Bias quantization off\n')
        
        if config.error_model != 'none' and config.error_model != 'generic':
            file.write('  Weight error on (' + config.error_model + ')\n')
        elif config.error_model == 'generic':
            if config.proportional_error:
                file.write('  Programming error (proportional): {:.3f}%\n'.format(100*config.alpha_error))
            else:
                file.write('  Programming error (uniform): {:.3f}%\n'.format(100*config.alpha_error))
        else:
            file.write('  Programming error off\n')
        if config.noise_model != "none":
            if config.noise_model == "generic":
                if config.alpha_noise > 0:
                    if config.proportional_noise:
                        file.write('  Read noise (proportional): {:.3f}%\n'.format(100*config.alpha_noise))
                    else:
                        file.write('  Read noise (uniform): {:.3f}%\n'.format(100*config.alpha_noise))
                else:
                    file.write('  Read noise off\n')
            else:
                file.write('  Read noise on (' + config.noise_model + ')\n')
        else:
            file.write('  Read noise off\n')
        if config.adc_bits > 0:
            if config.ADC_per_ibit:
                file.write('    ADC per input bit: ON\n')
            file.write('    ADC range option: ' + config.adc_range_option + "\n")
            if config.Nslices > 1 and config.adc_range_option == "calibrated":
                file.write('    Bit sliced ADC range calibration percentile: ' + str(config.pct) + '%\n')
            file.write('    ADC topology: ' + str(config.adc_type) + "\n")
        else:
            file.write('  ADC off\n')
        if config.dac_bits > 0:
            if np.min(config.dac_bits_vec) == np.max(config.dac_bits_vec):
                file.write('  Activation quantization on, ' + str(config.dac_bits) + ' bits\n')
            else:
                file.write('  Activation quantization on, variable: ' + str(np.min(config.dac_bits_vec)) + '-' + str(np.max(config.dac_bits_vec)) + ' bits\n')
            file.write('  Input bit slicing: ' + str(config.input_bitslicing) + "\n")
            if config.input_bitslicing:
                file.write('     Input slice size: ' + str(config.input_slice_size) + " bits\n")
        else:
            file.write('  Activation quantization off\n')
        if config.Icol_max > 0:
            file.write('  Column current clipping on: {:.3f} uA\n'.format(config.Icol_max*1e6))
        if config.Rp_col > 0:
            file.write('  Rp column = {:.3e} (ohms)\n'.format(config.Rp_col))
        if config.Rp_row > 0 and not config.gate_input:
            file.write('  Rp row = {:.3e} (ohms)\n'.format(config.Rp_row))
        if config.Rp_col == 0 and (config.Rp_row == 0 or config.gate_input):
            file.write('  Parasitics off\n')
        if config.Rp_col > 0 or config.Rp_row > 0:
            file.write('  Gate input mode: ' + str(config.gate_input) + "\n")
        if config.infinite_on_off_ratio:
            file.write('  On off ratio: infinite\n')
        else:
            file.write('  On off ratio: {:.1f}\n'.format(config.Rmax/config.Rmin))
        if config.t_drift > 0 or config.drift_model != "none":
            file.write('  Weight drift on, ' + str(config.t_drift) + ' days\n')
            file.write('  Drift model: ' + config.drift_model + "\n")
        else:
            file.write('  Weight drift off\n')
        if config.useGPU:
            file.write('  GPU: ' + str(config.gpu_num) + "\n")
        file.write("=======================================\n")


def inverse_minmax_latency(value, max, min):
    #max_, min_ 값 찾아야함
    newmax = 100
    newmin = 0
    max_ = max
    min_ = min
    return value*(max_ - min_)/(newmax-newmin) + min_

class RegressionModel_latency(nn.Module):
    def __init__(self):
        super(RegressionModel_latency, self).__init__()
        self.fc1 = nn.Linear(7,16)
        self.bn1 = nn.BatchNorm1d(num_features = 16)
        self.fc2 = nn.Linear(16,32)
        self.bn2 = nn.BatchNorm1d(num_features = 32)
        self.fc3 = nn.Linear(32,64)
        self.bn3 = nn.BatchNorm1d(num_features = 64)
        self.fc4 = nn.Linear(64, 128)
        self.bn4 = nn.BatchNorm1d(num_features = 128)
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(num_features = 64)
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(num_features = 32)
        self.fc7 = nn.Linear(32, 16)  # Output layer with 4 neurons for 4 output parameters
        self.bn7 = nn.BatchNorm1d(num_features = 16)
        self.fc8 = nn.Linear(16, 1)
        # self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.fc8(x)
        return x

class RegressionModel_power(nn.Module):
    def __init__(self):
        super(RegressionModel_power, self).__init__()
        self.fc1 = nn.Linear(7,128)
        self.bn1 = nn.BatchNorm1d(num_features = 128)
        self.fc2 = nn.Linear(128,128)
        self.bn2 = nn.BatchNorm1d(num_features = 128)
        self.fc3 = nn.Linear(128,64)
        self.bn3 = nn.BatchNorm1d(num_features = 64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(num_features = 32)
        self.fc5 = nn.Linear(32, 16)  # Output layer with 4 neurons for 4 output parameters
        self.bn5 = nn.BatchNorm1d(num_features = 16)
        self.fc6 = nn.Linear(16, 1)
        # self.dropout = nn.Dropout(0.1)
        self.relu = nn.ELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.fc6(x)
        return x
 
scaler_x_params_latency = joblib.load(f"{navcim_dir}/Inference_pytorch/predict_model/scaler_x_params_latency_noPGS.pkl")
scaler_y_params_latency = joblib.load(f"{navcim_dir}/Inference_pytorch/predict_model/scaler_y_params_latency_noPGS.pkl")
scaler_x_latency = StandardScaler()
scaler_x_latency.mean_ = scaler_x_params_latency['mean']
scaler_x_latency.scale_ = scaler_x_params_latency['scale']

scaler_x_params_power = joblib.load(f"{navcim_dir}/Inference_pytorch/predict_model/scaler_x_params_power.pkl")
scaler_x_power = StandardScaler()
scaler_x_power.mean_ = scaler_x_params_power['mean']
scaler_x_power.scale_ = scaler_x_params_power['scale']

loaded_latency_model = RegressionModel_latency()
loaded_power_model = RegressionModel_power()

device="cpu"
# Load the saved model state dictionary
loaded_latency_model.load_state_dict(torch.load(f"{navcim_dir}/Inference_pytorch/predict_model/regression_model_latency_noPGS.pth",map_location=device))
loaded_power_model.load_state_dict(torch.load(f"{navcim_dir}/Inference_pytorch/predict_model/regression_model_power.pth",map_location=device))

# Set the model to evaluation mode
loaded_latency_model = loaded_latency_model.to(device)
loaded_power_model = loaded_power_model.to(device)
loaded_latency_model.eval()
loaded_power_model.eval()

#area
loaded_model_filename_area = f'{navcim_dir}/Inference_pytorch/predict_model/mlp_area_predict.pkl'
loaded_scaler_filename_area = f'{navcim_dir}/Inference_pytorch/predict_model/mlp_scaler_x_area.pkl'
loaded_scaler_y_filename_area = f'{navcim_dir}/Inference_pytorch/predict_model/mlp_scaler_y_area.pkl'

with open(loaded_model_filename_area, 'rb') as file:
    loaded_pred_area = joblib.load(file)

scaler_x_scaler_param_area = joblib.load(loaded_scaler_filename_area)

scaler_x_scaler_area = StandardScaler()
scaler_x_scaler_area.mean_ = scaler_x_scaler_param_area['mean']
scaler_x_scaler_area.scale_ = scaler_x_scaler_param_area['scale']

scaler_y_scaler_param_area = joblib.load(loaded_scaler_y_filename_area)

scaler_y_scaler_area = StandardScaler()
scaler_y_scaler_area.mean_ = scaler_y_scaler_param_area['mean']
scaler_y_scaler_area.scale_ = scaler_y_scaler_param_area['scale']

#leakage_power
loaded_model_filename_leakage = f'{navcim_dir}/Inference_pytorch/predict_model/mlp_leakage_power_predict.pkl'
loaded_scaler_filename_leakage = f'{navcim_dir}/Inference_pytorch/predict_model/mlp_scaler_x_leakage_power.pkl'
loaded_scaler_y_filename_leakage = f'{navcim_dir}/Inference_pytorch/predict_model/mlp_scaler_y_leakage_power.pkl'

with open(loaded_model_filename_leakage, 'rb') as file:
    loaded_pred_leakage = joblib.load(file)

scaler_x_scaler_param_leakage = joblib.load(loaded_scaler_filename_leakage)

scaler_x_scaler_leakage = StandardScaler()
scaler_x_scaler_leakage.mean_ = scaler_x_scaler_param_leakage['mean']
scaler_x_scaler_leakage.scale_ = scaler_x_scaler_param_leakage['scale']

scaler_y_scaler_param_leakage = joblib.load(loaded_scaler_y_filename_leakage)

scaler_y_scaler_leakage = StandardScaler()
scaler_y_scaler_leakage.mean_ = scaler_y_scaler_param_leakage['mean']
scaler_y_scaler_leakage.scale_ = scaler_y_scaler_param_leakage['scale']

def config_to_identifier(config):
    return tuple(tuple(subconfig) for subconfig in config)

def find_common_configs(config_orders):
    # convert each model's configuration to an identifier
    all_configs = [set(config_to_identifier(config) for config in model_configs) for model_configs in config_orders.values()]
    
    # find common configuration identifiers
    common_config_identifiers = set.intersection(*all_configs)
    return common_config_identifiers

def filter_configs(config_orders, common_config_identifiers):
    for model in config_orders.keys():
        # convert configurations in the current model to identifiers and filter only common configurations
        config_orders[model] = [config for config in config_orders[model] if config_to_identifier(config) in common_config_identifiers]

def is_non_dominated(current_point, other_points):
    for other in other_points:
      if all(o >= c for o, c in zip(other, current_point)) and any(o > c for o, c in zip(other, current_point)):
          return 0
    return 1

def check_is_map(tuples):
  combinations = itertools.product(*tuples)
  tile_map_set_tmp = []
  
  for combination in combinations:
    num = [[] for i in range(len(model_list))]
    try:
      for i in combination[0][0]:
        for j in range(len(model_list)):
          num[j].append(tile_num_table[j][config_zip.index(i)])
    except Exception as e:
      error_info = traceback.format_exc()
      print(error_info)

    # tile mappability comparison code
    check_table = []
    try:
      for index, conf_tile  in enumerate(combination):
        calculate = [0 for i in range(args.heterogeneity)]
        for i,t in enumerate(conf_tile[1]):
          calculate[int(t)] += num[index][int(t)][i]
        check_table.append(calculate)
    except Exception as e:
      error_info = traceback.format_exc()
      print(error_info)

    non_dominated_table = [is_non_dominated(point, [p for p in check_table if p != point]) for point in check_table]
    non_dominated_count = sum(non_dominated_table)
    
    if non_dominated_count == 1:
      dominate_point_location = non_dominated_table.index(1)
      nondominate_point = combination[dominate_point_location] + (1,)
      combination = combination[:dominate_point_location] + (nondominate_point,) + combination[dominate_point_location+1:]
      tile_map_set_tmp.append(combination)
  
  tile_mapping_set_tmp.append(tile_map_set_tmp)

def sort_key(path):
  # 파일명에서 숫자를 추출
  numbers = re.findall(r'ADC:(\d+)_Cellbit:(\d+)_SA_row:(\d+)_SA_col:(\d+)_PE:(\d+)_TL:(\d+)', path)
  if numbers:
      return tuple(map(int, numbers[0]))
  return (0, 0, 0, 0)  

def read_and_sort_file(file_path):
  with open(file_path, 'r') as file:
    lines = file.readlines()
    data = []
    check = []
    for line in lines:
      try:
        parts = line.split('"')[1].split('_')[0]
      except Exception as e:
        print(line)


      
      tile_selection = line.split('"')[1].replace(parts+'_','')
      if parts not in check:
        data.append((eval(parts),tile_selection.split('_')))
        check.append(parts)
  return data

def crossover(parent1, parent2, individual_length):

  point = random.randint(1, individual_length - 1)
 
  return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(individual):
  return [gene if random.random() > args.mutation_rate else str((int(gene) + 1) % args.heterogeneity) for gene in individual]

def generate_population_with_crossover_and_mutation(base_individual, size):
  population = []
  while len(population) < size:

    parent1 = mutate(base_individual[:])
    parent2 = mutate(base_individual[:])

    child1, child2 = crossover(parent1, parent2, len(base_individual))

    child1 = mutate(child1)
    child2 = mutate(child2)

    population.extend([child1, child2])
  
  return population[:size]

# GA
def run_ga_first_generation(config):
    population = generate_population_with_crossover_and_mutation(config[1], args.population_size) #args.population_size
    return population

def run_ga_after_first_generation(config):
  if(len(config)>=2):
    next_generation = []
    while len(next_generation) < args.population_size:
      parent1, parent2 = random.sample(config, 2)
      child1, child2 = crossover(parent1, parent2, len(config[0]))
      child1 = mutate(child1)
      child2 = mutate(child2)
      next_generation.extend([child1, child2])
    population = next_generation[:args.population_size]
  elif(len(config)==1) and config[0] != []:
    try:
      population = generate_population_with_crossover_and_mutation(config[0], args.population_size)
    except Exception as e:
      print(e, config[0])
  else:
    population = []
  return population

def injection_rate(activation_size, Nbit, FPS, bus_width, freq):
  rate = (activation_size * Nbit * FPS) / (bus_width * freq)
  return rate

def flit_cycle(unit_rep,unit_wire,tile_width,clk,minDist):
  numRepeater = math.ceil(tile_width/minDist)
  if numRepeater>0:
    return math.ceil((unit_rep) * tile_width * clk)
  else:
    return math.ceil((unit_wire) * tile_width * clk)

def get_divisor(n ,minimum):
  data = []
  for i in range(1, n // 2 + 1):
      if n % i == 0:
          if i>=minimum:
            data.append(i)
  data.append(n)
  return data


def distribute(row,col,data_in,data_out, CLUSTER, rest_tile_list, rest_cluster):
  try:
    data_intra_spread = data_in/row
    data_intra_gather = data_out/col

    minimum_split_row=math.ceil(row/CLUSTER)
    minimum_split_col=math.ceil(col/CLUSTER)
    candidate_row= get_divisor(row, minimum_split_row)
    candidate_col= get_divisor(col, minimum_split_col)
    result= []
    final_cost=[]

    for split_row in candidate_row:
      for split_col in candidate_col:
        num_of_tile_in_cluster = (row/split_row) * (col/split_col)
        cost = data_in*split_col+(data_intra_gather+data_intra_spread)*num_of_tile_in_cluster+data_out*split_row
        # if final_cost is None or cost < final_cost:
        final_cost.append(cost)
        result.append([split_row,split_col])

    sorted_indices = sorted(range(len(final_cost)), key=lambda k: final_cost[k])

    if rest_tile_list == None:
      return result[sorted_indices[0]]
    
    final_result = None
    
    for idx in sorted_indices :
      number_of_cluster = result[idx][0] * result[idx][1]
      num_tile_per_cluster = (row/result[idx][0]) *(col/result[idx][1])
      success = []
      rest_tile_tmp = rest_tile_list.copy()
      rest_cluster_tmp = rest_cluster
      for n_c in range(number_of_cluster):
        is_rest = 0
        for i,rest in enumerate(rest_tile_tmp):
          if rest > num_tile_per_cluster :
            is_rest =1
            del rest_tile_tmp[i]
            success.append(1)
            break

        if is_rest == 0:
          if rest_cluster_tmp == 0 :
            success.append(0)

          success.append(1)
          rest_cluster_tmp -= 1
      

      if 0 not in success:
        final_result = result[idx]
        break
    return final_result
  except Exception as e:
    error_info = traceback.format_exc()
    print(e,"distribute")
    print(error_info)
         
   
def make_file_name(config):
  file_name = ""
  for i in config:
    file_name += f"ADC{i[4]}_Cellbit{i[5]}_SArow{i[0]}_SAcol{i[1]}_PE{i[2]}_TL{i[3]}_"
  return file_name
##place and rounting
def profile_for_select(i , tmp_dfg, Nbit ,Bus_width, chip_clk_freq, cluster_clk_freq, FPS ,CLUSTER, iter, cluster_wr, element_wr, shape, level, count, chip_width, chip_number,tile_grid, mapping_info):
  row = int(shape[2])
  col = int(shape[3])

  num_of_row = math.ceil(row/int(CLUSTER))
  num_of_col = math.ceil(col/int(CLUSTER))
  number_of_cluster = num_of_row * num_of_col

  key_list=list(tmp_dfg.keys())
  numBitToLoadIn=None
  numBitToLoadOut=None
  numInVector = None
  before_node_list =[]
  
  kernel_size = 1
  stride = 1
  if tmp_dfg[str(i)][6] != "":
    kernel_size = int(tmp_dfg[str(i)][6])
  if tmp_dfg[str(i)][7] != "":
    stride= int(tmp_dfg[str(i)][7])
    
  For_dense = None
  if i=='0':
    numInVector = (224-kernel_size+1)/stride * (224-kernel_size+1)/stride
    if model == "VGG8" or model == "DenseNet40":
      numInVector = (32-kernel_size+1)/stride * (32-kernel_size+1)/stride
    numBitToLoadIn = numInVector * Nbit *  kernel_size* kernel_size* 3
  else:
      for key_iter_for_fing_dest,p in enumerate(tmp_dfg.keys()):
          if tmp_dfg[p][1]==i or tmp_dfg[p][2]==i:
            if tmp_dfg[str(i)][0] =="nn.dense":
              For_dense = int(tmp_dfg[p][3][1])
            else:
              numInVector = (int(tmp_dfg[p][3][2])-kernel_size+1)/stride * (int(tmp_dfg[p][3][3])-kernel_size+1)/stride
              numBitToLoadIn = numInVector * Nbit * kernel_size* kernel_size *int(tmp_dfg[p][3][1])
            before_node_list.append(int(p))
              
  
  if tmp_dfg[str(i)][0] =="nn.dense":
    numInVector = 1
    numBitToLoadIn = numInVector * Nbit * For_dense
  injection = injection_rate(numBitToLoadIn/num_of_row, Nbit, FPS, Bus_width, cluster_clk_freq)

  numBitToLoadOut = numInVector * Nbit * int(tmp_dfg[str(i)][3][1])
  tmp_dfg[str(i)][4] = numBitToLoadOut
  
  addition_cluster = 0
  for r in range(number_of_cluster):
    exist = 0
    for t in mapping_info.keys():
      if (i not in mapping_info[t][0]) and (mapping_info[t][1] == chip_number) and (mapping_info[t][3] >= int(row/math.ceil(row/CLUSTER))*int(col/math.ceil(col/CLUSTER))):
        mapping_info[t][0].append(i)
        exist = 1
        target= t.split("-")
        offset =int(CLUSTER**2)- mapping_info[t][3]
        node_in_tile =  mapping_info[t][2]
        break
    if exist == 0:
      tile_grid[iter[0],iter[1]] = i
      if (level+1)*(chip_width) <= count:
        iter[1]+=1
        level+=1
      else:
        if level%2==0:
          iter[0]+=1
        else:
          iter[0]-=1
      target= iter
      offset = 0
      node_in_tile = np.full(int(CLUSTER**2),-1)
      count+=1
      addition_cluster += 1
   
    
    cluster_wr.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{target[0]}-{target[1]}",f"chip{chip_number}",numBitToLoadIn/num_of_row,injection,num_of_row,num_of_col])
   
    name_of_element_tile=f"{i}_{r}"
    element_tile1=np.full(int(CLUSTER**2),-1)

    for element in range(int(row/math.ceil(row/CLUSTER))*int(col/math.ceil(col/CLUSTER))):
      element_tile1[offset+element]=offset+element
      node_in_tile[offset+element]=i
    
    tmp_str1 = ""
    for li in element_tile1:
      tmp_str1= tmp_str1 +str(li) +" "
    injection_element_send = injection_rate(numBitToLoadIn/row, Nbit, FPS, Bus_width, chip_clk_freq)
    injection_element_receive = injection_rate(numBitToLoadOut/col, Nbit, FPS, Bus_width, chip_clk_freq)
    element_wr.writerow([name_of_element_tile,tmp_str1,(numBitToLoadIn/row),injection_element_send,(numBitToLoadOut/col),injection_element_receive,row,col])
    
    
    if exist == 1:
      mapping_info[f"{target[0]}-{target[1]}"][2] = node_in_tile
      mapping_info[f"{target[0]}-{target[1]}"][3] = mapping_info[f"{target[0]}-{target[1]}"][3] - int(row/math.ceil(row/CLUSTER))*int(col/math.ceil(col/CLUSTER))
    else:
      mapping_info[f"{target[0]}-{target[1]}"]=[[i],chip_number,node_in_tile,(CLUSTER**2)-int(row/math.ceil(row/CLUSTER))*int(col/math.ceil(col/CLUSTER))]
  return iter, count, level, tile_grid, num_of_col ,addition_cluster
  #----

def profile_for_select_distribute(i , tmp_dfg, Nbit ,Bus_width, chip_clk_freq, cluster_clk_freq, FPS ,CLUSTER, iter, cluster_wr,element_wr, shape, level, count, chip_width,chip_number,tile_grid, mapping_info, placement_list):
  row = int(shape[2])
  col = int(shape[3])
  numBitToLoadIn=None
  numBitToLoadOut=None
  numInVector = None
  key_list=list(tmp_dfg.keys())
  before_node_list =[]

  kernel_size = 1
  stride = 1
  if tmp_dfg[str(i)][6] != "":
    kernel_size = int(tmp_dfg[str(i)][6])
  if tmp_dfg[str(i)][7] != "":
    stride= int(tmp_dfg[str(i)][7])
    
  For_dense = None
  if i=='0':
    numInVector = (224-kernel_size+1)/stride * (224-kernel_size+1)/stride
    if model == "VGG8" or model == "DenseNet40":
      numInVector = (32-kernel_size+1)/stride * (32-kernel_size+1)/stride
    numBitToLoadIn = numInVector * Nbit *  kernel_size* kernel_size* 3
  else:
    for key_iter_for_fing_dest,p in enumerate(tmp_dfg.keys()):
      if tmp_dfg[p][1]==i or tmp_dfg[p][2]==i:
        if tmp_dfg[str(i)][0] =="nn.dense":
          For_dense = int(tmp_dfg[p][3][1])
        else:
          numInVector = (int(tmp_dfg[p][3][2])-kernel_size+1)/stride * (int(tmp_dfg[p][3][3])-kernel_size+1)/stride
          numBitToLoadIn = numInVector * Nbit * kernel_size* kernel_size *int(tmp_dfg[p][3][1])
        before_node_list.append(int(p))
  
  if tmp_dfg[str(i)][0] =='nn.dense':
    numInVector = 1
    numBitToLoadIn = numInVector * Nbit * For_dense
    
  numBitToLoadOut = numInVector * Nbit * int(tmp_dfg[str(i)][3][1])
  tmp_dfg[str(i)][4] = numBitToLoadOut
  
  rest_tile_in_cluster =[]
  for t in mapping_info.keys():
    if mapping_info[t][1] == chip_number:
      rest_tile_in_cluster.append(mapping_info[t][3])

  if placement_list == None:
    split = distribute(row,col,numBitToLoadIn,numBitToLoadOut,CLUSTER,None,None)
  else:
    split = distribute(row,col,numBitToLoadIn,numBitToLoadOut,CLUSTER,rest_tile_in_cluster,len(placement_list[chip_number]))
    if split == None:
      return None, None, None, None, None ,None, None

  num_of_row = split[0]
  num_of_col = split[1]
  number_of_cluster = num_of_row * num_of_col
  num_tile_per_cluster = (row/num_of_row) *(col/num_of_col)
  
  addition_cluster = 0
  injection = injection_rate(numBitToLoadIn/num_of_row, Nbit, FPS, Bus_width, cluster_clk_freq)
  if placement_list == None:
    for r in range(number_of_cluster):
      exist = 0
      for t in mapping_info.keys():
        if (i not in mapping_info[t][0]) and (mapping_info[t][1] == chip_number) and (mapping_info[t][3] >= int(num_tile_per_cluster)):
          mapping_info[t][0].append(i)
          exist = 1
          target= t.split("-")
          offset =int(CLUSTER**2)- mapping_info[t][3]
          node_in_tile =  mapping_info[t][2]
          break
      if exist == 0:
        if iter[1]> chip_width:
          print(f"error_{mapping_info}_{chip_width}") 
        tile_grid[iter[0],iter[1]] = i
        if (level+1)*(chip_width) <= count:
          iter[1]+=1
          level+=1
        else:
          if level%2==0:
            iter[0]+=1
          else:
            iter[0]-=1
        count+=1
        addition_cluster += 1
        target= iter
        offset = 0
        node_in_tile = np.full(int(CLUSTER**2),-1)

      
      cluster_wr.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{target[0]}-{target[1]}",f"chip{chip_number}",numBitToLoadIn/num_of_row,injection,num_of_row,num_of_col])

      name_of_element_tile=f"{i}_{r}"
      element_tile1=np.full(int(CLUSTER**2),-1)
      
      for element in range(int(num_tile_per_cluster)):
        element_tile1[offset+element]=offset+element
        node_in_tile[offset+element]=i
        
      tmp_str1 = ""
      for li in element_tile1:
        tmp_str1= tmp_str1 +str(li) +" "

      injection_element_send = injection_rate(numBitToLoadIn/row, Nbit, FPS, Bus_width, chip_clk_freq)
      injection_element_receive = injection_rate(numBitToLoadOut/col, Nbit, FPS, Bus_width, chip_clk_freq)
      element_wr.writerow([name_of_element_tile,tmp_str1,(numBitToLoadIn/row),injection_element_send,(numBitToLoadOut/col),injection_element_receive,row,col])

      if exist == 1:
        mapping_info[f"{target[0]}-{target[1]}"][2] = node_in_tile
        mapping_info[f"{target[0]}-{target[1]}"][3] = mapping_info[f"{target[0]}-{target[1]}"][3] - int(num_tile_per_cluster)
      else:
        mapping_info[f"{target[0]}-{target[1]}"]=[[i],chip_number,node_in_tile,(CLUSTER**2)-int(num_tile_per_cluster)]
  else:
    for r in range(number_of_cluster):
      exist = 0
      for t in mapping_info.keys():
        if (i not in mapping_info[t][0]) and (mapping_info[t][1] == chip_number) and (mapping_info[t][3] >= int(num_tile_per_cluster)):
          mapping_info[t][0].append(i)
          exist = 1
          target= t.split("-")
          offset =int(CLUSTER**2)- mapping_info[t][3]
          node_in_tile =  mapping_info[t][2]
          break
      if exist == 0:
        target=placement_list[chip_number].pop(0).split("-")
        iter = target
        count+=1
        addition_cluster += 1
        offset = 0
        node_in_tile = np.full(int(CLUSTER**2),-1)

      
      cluster_wr.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{target[0]}-{target[1]}",f"chip{chip_number}",numBitToLoadIn/num_of_row,injection,num_of_row,num_of_col])

      name_of_element_tile=f"{i}_{r}"
      element_tile1=np.full(int(CLUSTER**2),-1)
      
      for element in range(int(num_tile_per_cluster)):
        element_tile1[offset+element]=offset+element
        node_in_tile[offset+element]=i
        
      tmp_str1 = ""
      for li in element_tile1:
        tmp_str1= tmp_str1 +str(li) +" "

      injection_element_send = injection_rate(numBitToLoadIn/row, Nbit, FPS, Bus_width, chip_clk_freq)
      injection_element_receive = injection_rate(numBitToLoadOut/col, Nbit, FPS, Bus_width, chip_clk_freq)
      element_wr.writerow([name_of_element_tile,tmp_str1,(numBitToLoadIn/row),injection_element_send,(numBitToLoadOut/col),injection_element_receive,row,col])

      if exist == 1:
        mapping_info[f"{target[0]}-{target[1]}"][2] = node_in_tile
        mapping_info[f"{target[0]}-{target[1]}"][3] = mapping_info[f"{target[0]}-{target[1]}"][3] - int(num_tile_per_cluster)
      else:
        mapping_info[f"{target[0]}-{target[1]}"]=[[i],chip_number,node_in_tile,(CLUSTER**2)-int(num_tile_per_cluster)]
  
  return iter, count, level, tile_grid, num_of_col ,addition_cluster, placement_list

def profile_for_select_distribute_predict(i , tmp_dfg, Nbit ,Bus_width, chip_clk_freq, cluster_clk_freq, FPS ,CLUSTER, iter, cluster_wr,element_wr, shape, level, count, chip_width,chip_number,tile_grid, mapping_info, placement_list,mapping_num_list,model):
  try:
    row = int(shape[2])
    col = int(shape[3])
    numBitToLoadIn=None
    numBitToLoadOut=None
    numInVector = None
    key_list=list(tmp_dfg.keys())
    before_node_list =[]

    list_for_predict = {}
    
    kernel_size = 1
    stride = 1
    if tmp_dfg[str(i)][6] != "":
      kernel_size = int(tmp_dfg[str(i)][6])
    if tmp_dfg[str(i)][7] != "":
      stride= int(tmp_dfg[str(i)][7])
      
    For_dense = None
    if i=='0':
      numInVector = (224-kernel_size+1)/stride * (224-kernel_size+1)/stride
      if model == "VGG8" or model == "DenseNet40":
        numInVector = (32-kernel_size+1)/stride * (32-kernel_size+1)/stride
      numBitToLoadIn = numInVector * Nbit *  kernel_size* kernel_size* 3
    else:
      for key_iter_for_fing_dest,p in enumerate(tmp_dfg.keys()):
        if tmp_dfg[p][1]==i or tmp_dfg[p][2]==i:
          if tmp_dfg[str(i)][0] =="nn.dense":
            For_dense = int(tmp_dfg[p][3][1])
          else:
            numInVector = (int(tmp_dfg[p][3][2])-kernel_size+1)/stride * (int(tmp_dfg[p][3][3])-kernel_size+1)/stride
            numBitToLoadIn = numInVector * Nbit * kernel_size* kernel_size *int(tmp_dfg[p][3][1])
          before_node_list.append(int(p))
    
    if tmp_dfg[str(i)][0] =='nn.dense':
      numInVector = 1
      numBitToLoadIn = numInVector * Nbit * For_dense
      
    numBitToLoadOut = numInVector * Nbit * int(tmp_dfg[str(i)][3][1])
    tmp_dfg[str(i)][4] = numBitToLoadOut

    rest_tile_in_cluster =[]
    for t in mapping_info.keys():
      if mapping_info[t][1] == chip_number:
        rest_tile_in_cluster.append(mapping_info[t][3])

    if placement_list == None:
      split = distribute(row,col,numBitToLoadIn,numBitToLoadOut,CLUSTER,None,None)
    else:
      split = distribute(row,col,numBitToLoadIn,numBitToLoadOut,CLUSTER,rest_tile_in_cluster,len(placement_list[chip_number]))
      if split == None:
        return None, None, None, None, None ,None, None, None, None

    num_of_row = split[0]
    num_of_col = split[1]
    number_of_cluster = num_of_row * num_of_col
    num_tile_per_cluster = (row/num_of_row) *(col/num_of_col)
    
    addition_cluster = 0
    injection = injection_rate(numBitToLoadIn/num_of_row, Nbit, FPS, Bus_width, cluster_clk_freq)
    mapping_num_list.append(int(number_of_cluster))
    
    if placement_list == None:
      for r in range(number_of_cluster):
        exist = 0
        for t in mapping_info.keys():
          if (i not in mapping_info[t][0]) and (mapping_info[t][1] == chip_number) and (mapping_info[t][3] >= int(num_tile_per_cluster)):
            mapping_info[t][0].append(i)
            exist = 1
            target= t.split("-")
            offset =int(CLUSTER**2)- mapping_info[t][3]
            node_in_tile =  mapping_info[t][2]
            break
        if exist == 0:
          if iter[1]> chip_width:
            print(f"error_{mapping_info}_{chip_width}") 
          tile_grid[iter[0],iter[1]] = i
          if (level+1)*(chip_width) <= count:
            iter[1]+=1
            level+=1
          else:
            if level%2==0:
              iter[0]+=1
            else:
              iter[0]-=1
          count+=1
          addition_cluster += 1
          target= iter
          offset = 0
          node_in_tile = np.full(int(CLUSTER**2),-1)

        name_of_element_tile=f"{i}_{r}"
        element_tile1=np.full(int(CLUSTER**2),-1)
        
        for element in range(int(num_tile_per_cluster)):
          element_tile1[offset+element]=offset+element
          node_in_tile[offset+element]=i
          
        tmp_str1 = ""
        for li in element_tile1:
          tmp_str1= tmp_str1 +str(li) +" "

        injection_element_send = injection_rate(numBitToLoadIn/row, Nbit, FPS, Bus_width, chip_clk_freq)
        injection_element_receive = injection_rate(numBitToLoadOut/col, Nbit, FPS, Bus_width, chip_clk_freq)

        if exist == 1:
          mapping_info[f"{target[0]}-{target[1]}"][2] = node_in_tile
          mapping_info[f"{target[0]}-{target[1]}"][3] = mapping_info[f"{target[0]}-{target[1]}"][3] - int(num_tile_per_cluster)
        else:
          mapping_info[f"{target[0]}-{target[1]}"]=[[i],chip_number,node_in_tile,(CLUSTER**2)-int(num_tile_per_cluster)]
      
      st1 = node[model][i][1]
      st2 = node[model][i][2]
      
      key_position1 = node_keys_list[model].index(st1) if st1 in node_keys_list[model] else None
      key_position2 = node_keys_list[model].index(st2) if st2 in node_keys_list[model] else None
      
      input_cluster = 0
      if key_position1 is not None:
        input_cluster += mapping_num_list[key_position1]
      if key_position2 is not None:
        input_cluster += mapping_num_list[key_position2]

      list_for_predict = {}
      list_for_predict["injection_cluster"] = injection
      list_for_predict["injection_element_send"] = injection_element_send
      list_for_predict["injection_element_receive"] = injection_element_receive
      
      list_for_predict["activation_size_cluster"] = numBitToLoadIn/num_of_row
      list_for_predict["activation_size_element_send"] = numBitToLoadIn/row
      list_for_predict["activation_size_element_receive"] = numBitToLoadOut/col

      if len(mapping_num_list)==1:
        list_for_predict["num_of_input_cluster"] = 1
      else:
        list_for_predict["num_of_input_cluster"] =  input_cluster
      list_for_predict["num_of_input_element_send"] = 1
      list_for_predict["num_of_input_element_receive"] = num_tile_per_cluster

      list_for_predict["num_of_dest_cluster"] = mapping_num_list[-1]
      list_for_predict["num_of_dest_element_send"] = num_tile_per_cluster
      list_for_predict["num_of_dest_element_receive"] = 1
    else:
      for r in range(number_of_cluster):
        exist = 0
        for t in mapping_info.keys():
          if (i not in mapping_info[t][0]) and (mapping_info[t][1] == chip_number) and (mapping_info[t][3] >= int(num_tile_per_cluster)):
            mapping_info[t][0].append(i)
            exist = 1
            target= t.split("-")
            offset =int(CLUSTER**2)- mapping_info[t][3]
            node_in_tile =  mapping_info[t][2]
            break
        if exist == 0:
          target=placement_list[chip_number].pop(0).split("-")
          iter = target
          count+=1
          addition_cluster += 1
          offset = 0
          node_in_tile = np.full(int(CLUSTER**2),-1)

        
        cluster_wr.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{target[0]}-{target[1]}",f"chip{chip_number}",numBitToLoadIn/num_of_row,injection,num_of_row,num_of_col])

        name_of_element_tile=f"{i}_{r}"
        element_tile1=np.full(int(CLUSTER**2),-1)
        
        for element in range(int(num_tile_per_cluster)):
          element_tile1[offset+element]=offset+element
          node_in_tile[offset+element]=i
          
        tmp_str1 = ""
        for li in element_tile1:
          tmp_str1= tmp_str1 +str(li) +" "

        injection_element_send = injection_rate(numBitToLoadIn/row, Nbit, FPS, Bus_width, chip_clk_freq)
        injection_element_receive = injection_rate(numBitToLoadOut/col, Nbit, FPS, Bus_width, chip_clk_freq)
        element_wr.writerow([name_of_element_tile,tmp_str1,(numBitToLoadIn/row),injection_element_send,(numBitToLoadOut/col),injection_element_receive,row,col])

        if exist == 1:
          mapping_info[f"{target[0]}-{target[1]}"][2] = node_in_tile
          mapping_info[f"{target[0]}-{target[1]}"][3] = mapping_info[f"{target[0]}-{target[1]}"][3] - int(num_tile_per_cluster)
        else:
          mapping_info[f"{target[0]}-{target[1]}"]=[[i],chip_number,node_in_tile,(CLUSTER**2)-int(num_tile_per_cluster)]
        
      st1 = node[model][i][1]
      st2 = node[model][i][2]

      key_position1 = node_keys_list[model].index(st1) if st1 in node_keys_list[model] else None
      key_position2 = node_keys_list[model].index(st2) if st2 in node_keys_list[model] else None

      input_cluster = 0
      if key_position1 is not None:
        input_cluster += mapping_num_list[key_position1]
      if key_position2 is not None:
        input_cluster += mapping_num_list[key_position2]
      # print(f"k1:{key_position1},k2:{key_position2},i:{i},input_cluster:{input_cluster},mapping_num_list:{mapping_num_list}")

      list_for_predict = {}
      list_for_predict["injection_cluster"] = injection
      list_for_predict["injection_element_send"] = injection_element_send
      list_for_predict["injection_element_receive"] = injection_element_receive
      
      list_for_predict["activation_size_cluster"] = numBitToLoadIn/num_of_row
      list_for_predict["activation_size_element_send"] = numBitToLoadIn/row
      list_for_predict["activation_size_element_receive"] = numBitToLoadOut/col

      if len(mapping_num_list)==1:
        list_for_predict["num_of_input_cluster"] = 1
      else:
        list_for_predict["num_of_input_cluster"] =  input_cluster
      list_for_predict["num_of_input_element_send"] = 1
      list_for_predict["num_of_input_element_receive"] = num_tile_per_cluster

      list_for_predict["num_of_dest_cluster"] = mapping_num_list[-1]
      list_for_predict["num_of_dest_element_send"] = num_tile_per_cluster
      list_for_predict["num_of_dest_element_receive"] = 1

    return iter, count, level, tile_grid, num_of_col ,addition_cluster, placement_list, list_for_predict, mapping_num_list
  except Exception as e:
    error_info = traceback.format_exc()
    print(error_info)
    print("profile_for_select_distribute_predict", mapping_info)

def predict_booksim(node, cluster_width, chip1_width, cluster_flit_cycle, chip1_flit_cycle ,model, select, cluster_meter, chip1_meter , list_for_predict, need_area_leak):  
  input_data_cluster = np.array([[cluster_width, cluster_flit_cycle, cluster_meter, list_for_predict["injection_cluster"], list_for_predict["activation_size_cluster"], list_for_predict["num_of_input_cluster"], list_for_predict["num_of_dest_cluster"]]]) 
  input_data_element_send = np.array([[chip1_width, chip1_flit_cycle, chip1_meter, list_for_predict["injection_element_send"], list_for_predict["activation_size_element_send"], list_for_predict["num_of_input_element_send"], list_for_predict["num_of_dest_element_send"]]]) 
  input_data_element_receive = np.array([[chip1_width, chip1_flit_cycle, chip1_meter, list_for_predict["injection_element_receive"], list_for_predict["activation_size_element_receive"], list_for_predict["num_of_input_element_receive"], list_for_predict["num_of_dest_element_receive"]]]) 
  
  latency_predict_cluster = 0
  power_predict_cluster = 0
  latency_predict_element_send = 0 
  power_predict_element_send = 0 
  latency_predict_element_receive = 0 
  power_predict_element_receive = 0
  
  input_data_cluster_power = np.array([[cluster_width, cluster_flit_cycle, cluster_meter, list_for_predict["injection_cluster"], list_for_predict["activation_size_cluster"], list_for_predict["num_of_input_cluster"], list_for_predict["num_of_dest_cluster"]]])
  input_data_element_send_power = np.array([[chip1_width, chip1_flit_cycle, chip1_meter, list_for_predict["injection_element_send"], list_for_predict["activation_size_element_send"], list_for_predict["num_of_input_element_send"], list_for_predict["num_of_dest_element_send"]]]) 
  input_data_element_receive_power  = np.array([[chip1_width, chip1_flit_cycle, chip1_meter, list_for_predict["injection_element_receive"], list_for_predict["activation_size_element_receive"], list_for_predict["num_of_input_element_receive"], list_for_predict["num_of_dest_element_receive"]]])

  with torch.no_grad():
    X_normalized_latency = scaler_x_latency.transform(input_data_cluster)
    latency_predict_normalized = loaded_latency_model(torch.tensor(X_normalized_latency,dtype=torch.float32))

    latency_predict_cluster = inverse_minmax_latency(latency_predict_normalized,scaler_y_params_latency['max'], scaler_y_params_latency['min'])[0][0]

    X_normalized_power = scaler_x_power.transform(input_data_cluster_power)
    power_predict_normalized = loaded_power_model(torch.tensor(X_normalized_power,dtype=torch.float32))
    power_predict_cluster = power_predict_normalized[0][0]
    

  if input_data_element_send[0][-1] > 1:
    with torch.no_grad():
      X_normalized_latency = scaler_x_latency.transform(input_data_element_send)
      latency_predict_normalized = loaded_latency_model(torch.tensor(X_normalized_latency,dtype=torch.float32))

      latency_predict_element_send = inverse_minmax_latency(latency_predict_normalized,scaler_y_params_latency['max'], scaler_y_params_latency['min'])[0][0]

      X_normalized_power = scaler_x_power.transform(input_data_element_send_power)
      power_predict_normalized = loaded_power_model(torch.tensor(X_normalized_power,dtype=torch.float32))
      power_predict_element_send = power_predict_normalized[0][0]
  
  if input_data_element_receive[0][-2] > 1 :
    with torch.no_grad():
      X_normalized_latency = scaler_x_latency.transform(input_data_element_receive)
      latency_predict_normalized = loaded_latency_model(torch.tensor(X_normalized_latency,dtype=torch.float32))

      latency_predict_element_receive = inverse_minmax_latency(latency_predict_normalized,scaler_y_params_latency['max'], scaler_y_params_latency['min'])[0][0]

      X_normalized_power = scaler_x_power.transform(input_data_element_receive_power)
      power_predict_normalized = loaded_power_model(torch.tensor(X_normalized_power,dtype=torch.float32))
      power_predict_element_receive = power_predict_normalized[0][0]

    
  latency_result = latency_predict_cluster 
  latency_result_cluster = latency_predict_element_send + latency_predict_element_receive

  power_result = power_predict_cluster 
  power_result_cluster = power_predict_element_send + power_predict_element_receive 

  if need_area_leak == 1:
    input_data_area_leak = np.array([[cluster_width, cluster_meter]])

    mlp_area_cluster = scaler_x_scaler_area.transform(input_data_area_leak)
    pred_area_cluster = loaded_pred_area.predict(mlp_area_cluster)
    cluster_booksim_area = scaler_y_scaler_area.inverse_transform(pred_area_cluster.reshape(-1,1))

    mlp_leak_cluster = scaler_x_scaler_leakage.transform(input_data_area_leak)
    pred_leak_cluster = loaded_pred_leakage.predict(mlp_leak_cluster)
    cluster_booksim_leakage = scaler_y_scaler_leakage.inverse_transform(pred_leak_cluster.reshape(-1,1))

    input_data_area_leak_chip = np.array([[chip1_width, chip1_meter]])

    mlp_area_chip = scaler_x_scaler_area.transform(input_data_area_leak_chip)
    pred_area_chip = loaded_pred_area.predict(mlp_area_chip)
    chip1_booksim_area = scaler_y_scaler_area.inverse_transform(pred_area_chip.reshape(-1,1))

    mlp_leak_chip = scaler_x_scaler_leakage.transform(input_data_area_leak_chip)
    pred_leak_chip = loaded_pred_leakage.predict(mlp_leak_chip)
    chip1_booksim_leakage = scaler_y_scaler_leakage.inverse_transform(pred_leak_chip.reshape(-1,1))
    if chip1_booksim_leakage[0][0] < 0:
      chip1_booksim_leakage[0][0] = 0
    if chip1_booksim_area[0][0] < 0:
      chip1_booksim_area[0][0] = 0
    
    return latency_result,power_result,latency_result_cluster,power_result_cluster, cluster_booksim_area[0][0]*1000000,chip1_booksim_area[0][0]*1000000,cluster_booksim_leakage[0][0],chip1_booksim_leakage[0][0]
  else:
    return latency_result,power_result,latency_result_cluster,power_result_cluster

def execute_booksim(node, cluster_width, chip1_width, cluster_flit_cycle, chip1_flit_cycle, model, select, cluster_meter, chip1_meter, chip_period,cluster_period, cluster_buswidth ,chip1_buswidth, file_name, depth, NUM_ROW_cluster, NUM_COL_cluster ,idx):
  cmd1 = f'{navcim_dir}/booksim2/src/booksim {navcim_dir}/booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {cluster_flit_cycle} {chip1_flit_cycle} {navcim_dir}/Inference_pytorch/record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{select}.txt {navcim_dir}/Inference_pytorch/record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/ELEMENT_{model}_{file_name}{select}.txt {cluster_meter} {chip1_meter} {cluster_buswidth} {chip1_buswidth} 0 na {node} 1 | egrep "taken|Total Power|Total Area|Total leak Power" > ./record_{model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/BOOKSIM_{model}_{file_name}{select}{node}.txt'
  cmd2 = f'{navcim_dir}/booksim2/src/booksim {navcim_dir}/booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {cluster_flit_cycle} {chip1_flit_cycle} {navcim_dir}/Inference_pytorch/record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{select}.txt {navcim_dir}/Inference_pytorch/record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/ELEMENT_{model}_{file_name}{select}.txt {cluster_meter} {chip1_meter} {cluster_buswidth} {chip1_buswidth} 0 na {node} 2 | egrep "taken|Total Power|Total Area|Total leak Power" >> ./record_{model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/BOOKSIM_{model}_{file_name}{select}{node}.txt'
  cmd3 = f'{navcim_dir}/booksim2/src/booksim {navcim_dir}/booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {cluster_flit_cycle} {chip1_flit_cycle} {navcim_dir}/Inference_pytorch/record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{select}.txt {navcim_dir}/Inference_pytorch/record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/ELEMENT_{model}_{file_name}{select}.txt {cluster_meter} {chip1_meter} {cluster_buswidth} {chip1_buswidth} 0 na {node} 3 | egrep "taken|Total Power|Total Area|Total leak Power" >> ./record_{model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/BOOKSIM_{model}_{file_name}{select}{node}.txt'
  try:
    output = subprocess.check_output(
        cmd1, stderr=subprocess.STDOUT, shell=True)
    
  except subprocess.CalledProcessError as exc:
      print("Error!!!!!", exc.returncode, exc.output, cmd1)
      print(file_name, select)
      return 10000000000000000,10000000000000000,100000000000000000,100000000000000000,100000000000000000,100000000000000000

  try:
    output = subprocess.check_output(
        cmd2, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as exc:
      print("Error!!!!!", exc.returncode, exc.output, cmd2)
      return 10000000000000000,10000000000000000,100000000000000000,100000000000000000,100000000000000000,100000000000000000

  try:
    output = subprocess.check_output(
        cmd3, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as exc:
      print("Error!!!!!", exc.returncode, exc.output, cmd3)
      return 10000000000000000,10000000000000000,100000000000000000,100000000000000000,100000000000000000,100000000000000000

  fname = f"./record_{model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/BOOKSIM_{model}_{file_name}{select}{node}.txt"
  latency_result = 0
  energy_result=0
  area_cluster=0
  area_chip = 0 
  cluster_leak_power = 0
  chip_leak_power = 0
  with open(fname) as f:
    lines = f.readlines()
    try:
      latency_result = int(lines[0].split("\n")[0].split(" ")[3])* cluster_period + int(lines[4].split("\n")[0].split(" ")[3])*chip_period + int(lines[8].split("\n")[0].split(" ")[3])*chip_period
      energy_result = float(lines[1].split("\n")[0].split(" ")[3])* int(lines[0].split("\n")[0].split(" ")[3])* cluster_period+ float(lines[5].split("\n")[0].split(" ")[3])*int(lines[4].split("\n")[0].split(" ")[3])*chip_period  + float(lines[9].split("\n")[0].split(" ")[3])*int(lines[8].split("\n")[0].split(" ")[3])*chip_period
      area_cluster = float(lines[2].split("\n")[0].split(" ")[3])*1000000
      area_chip = float(lines[6].split("\n")[0].split(" ")[3])*1000000
      cluster_leak_power = float(lines[3].split("\n")[0].split(" ")[4])
      chip_leak_power = float(lines[7].split("\n")[0].split(" ")[4])
    except Exception as e:
      print(e)
      print(lines)

  return latency_result,energy_result*1000,area_cluster,area_chip,cluster_leak_power,chip_leak_power


def initialization(config, depth, dfg,model):
  num_tile = {}
  FPS_tmp_dict ={}
  shape_tmp_dict  ={}
  PPA_tmp_dict  ={}
  file_name = make_file_name(config)
  for cf in range(len(config)):
    NUM1_ROW = config[cf][0]
    NUM1_COL = config[cf][1]
    PE1 = config[cf][2]
    Tile1 = config[cf][3]
    ADC = config[cf][4]
    Cellbit=config[cf][5]

    shape_tmp={}
    fname1 = f"./shape/shape_{model}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"
    with open(fname1) as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            l=l.replace("\n", "")
            shape_tmp[i]=l.split(',')

    CLUSTER1 = int(math.ceil((NUM1_ROW*Tile1)/(config[0][0]*config[0][3])))
    compute_PPA_tmp={}
    FPS_latency = 0
    for layer_num, layer in enumerate(All_layer):
      latency1 = LATENCY.get(f"{model}_{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")-inter_connect_LATENCY.get(f"{model}_{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
      energy1 = POWER.get(f"{model}_{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")-inter_connect_POWER.get(f"{model}_{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
      area1 = AREA.get(f"{model}_{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")-inter_connect_AREA.get(f"{model}_{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
      compute_PPA_tmp[layer_num] = [latency1,energy1,area1] 
      FPS_latency = FPS_latency + latency1*1e-9
      if layer_num in num_tile:
        num_tile[layer_num] = max(math.ceil(int(shape_tmp[layer_num][4])/(CLUSTER1**2)), num_tile[layer_num])
      else :
        num_tile[layer_num] = math.ceil(int(shape_tmp[layer_num][4]) / (CLUSTER1**2))
    shape_tmp_dict[cf] = shape_tmp
    PPA_tmp_dict[cf] = compute_PPA_tmp
    FPS_tmp_dict[cf] = 1/FPS_latency
    
    tail_distribute = ""

  if not os.path.isdir(f"./record_GA_{model}/CLUSTER/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}"):
    os.makedirs(f"./record_GA_{model}/CLUSTER/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}")
  if not os.path.isdir(f"./record_GA_{model}/ELEMENT/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}"):
    os.makedirs(f"./record_GA_{model}/ELEMENT/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}")
  
  cluster_f = open(f"./record_GA_{model}/CLUSTER/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}/CLUSTER_{model}_{file_name}{tail_distribute}.txt",'w', newline='')
  cluster_wr = csv.writer(cluster_f)
  cluster_wr.writerow(["node","destination1","destination2","op","location","type","activation_size","injection_rate"])
  cluster_f.close()

  element_f = open(f"./record_GA_{model}/ELEMENT/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}/ELEMENT_{model}_{file_name}{tail_distribute}.txt",'w', newline='')
  element_wr = csv.writer(element_f)
  element_wr.writerow(["node","used","activation_size","injection_rate"])
  element_f.close()

  if not os.path.isdir(f"./record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}"):
    os.makedirs(f"./record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}")
  if not os.path.isdir(f"./record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}"):
    os.makedirs(f"./record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}")
  
  cluster_f = open(f"./record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}/CLUSTER_{model}_{file_name}{tail_distribute}.txt",'w', newline='')
  cluster_wr = csv.writer(cluster_f)
  cluster_wr.writerow(["node","destination1","destination2","op","location","type","activation_size","injection_rate"])
  cluster_f.close()

  element_f = open(f"./record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}/ELEMENT_{model}_{file_name}{tail_distribute}.txt",'w', newline='')
  element_wr = csv.writer(element_f)
  element_wr.writerow(["node","used","activation_size","injection_rate"])
  element_f.close()

  tot_cluster_tile = sum(num_tile.values())
  if args.distribute == 1:
    chip_width = math.ceil(math.sqrt(tot_cluster_tile)) + 1
  else:
    chip_width = math.ceil(math.sqrt(tot_cluster_tile)) + 1

  FPS = sum(FPS_tmp_dict.values())/len(config)

  X = np.linspace(-1,-1,chip_width)
  Y = np.linspace(-1,-1,chip_width)
  tile_grid,tile_type = np.meshgrid(X,Y)
  iter=[-1,1]
  level = np.array([0], dtype=int)
  count = np.array([0], dtype=int)
  tmp_dfg = dfg
  mapping_info = {}
  conv_dense=0
  total_latency=0
  total_energy=0
  total_area=0
  total_leakage=0
  selected_list=[]

  return shape_tmp_dict ,PPA_tmp_dict ,chip_width,FPS,iter,level,count,tile_grid,mapping_info,conv_dense,total_latency,total_energy,total_area,total_leakage,selected_list

def make_args_simulator(ht,config,config_cluster,tmp_dfg,shape1,compute_PPA1,chip_width,FPS,iter,level,count,tile_grid,mapping_info,conv_dense,total_latency,total_energy,total_area,total_leakage,dfg_key,selected_list,file_name, depth, model, placement,idx):
  NUM1_ROW = config[0]
  NUM1_COL = config[1]
  PE1 = config[2]
  Tile1 = config[3]
  ADC=config[4]
  Cellbit=config[5]

  NUM_ROW_cluster = config_cluster[0]
  NUM_COL_cluster = config_cluster[1]
  PE_cluster = config_cluster[2]
  Tile_cluster = config_cluster[3]
  ADC_cluster=config_cluster[4]
  Cellbit_cluster=config_cluster[5]

  CLUSTER1 = int((NUM_ROW_cluster*Tile_cluster)/(NUM1_ROW*Tile1))
  chip1_flit_cycle = flit_cycle(unitLatencyRep.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"), unitLatencyWire.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"),tile_width_meter.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"),clk_frequency.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"),minDist.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"))
  chip1_clk_freq = clk_frequency.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  
  cluster_flit_cycle = flit_cycle(unitLatencyRep.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}"), unitLatencyWire.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}"),tile_width_meter.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}"),clk_frequency.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}"),minDist.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}"))
  cluster_meter = max(tile_width_meter.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")*CLUSTER1, tile_width_meter.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}"))
  cluster_clk_freq = clk_frequency.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}")
  
  chip1_meter = tile_width_meter.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  chip1_buswidth = busWidth.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  cluster_buswidth = chip1_buswidth
  chip1_period = clk_period.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  period = clk_period.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}")

  # total_area = 0
  # total_leakage =0
  chip1_area = (chip1_meter**2)*1e12
  chip1_leakage =0 
  if f"{model}_layer{conv_dense}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}" in leakage_POWER:
    chip1_leakage = leakage_POWER.get(f"{model}_layer{conv_dense}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")*1e-6
  chip1_booksim_area  = 0
  cluster_booksim_area =0
  chip1_booksim_leakage  = 0
  cluster_booksim_leakage = 0
  cluster_area = (cluster_meter**2)*1e12
  tail_select=""
  for selected in selected_list:
    tail_select = tail_select + str(selected)
  
  tail_distribute = tail_select

  Nbit=8
  node_col={}
  inverse = {}
  tail1=f"{tail_select}{ht}"
  
  for i in dfg_key: 
    if (tmp_dfg[str(i)][0]=='nn.conv2d' or tmp_dfg[str(i)][0]=='nn.dense'):
      os.makedirs(f"./record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}", exist_ok=True)
      os.makedirs(f"./record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}", exist_ok=True)
      os.makedirs(f"./record_{model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}", exist_ok=True)
      if depth == 0 :
        shutil.copy(f"./record_{model}/CLUSTER/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{model}_{file_name}{tail_distribute}.txt" ,f"./record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{tail1}.txt")
        shutil.copy(f"./record_{model}/ELEMENT/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{model}_{file_name}{tail_distribute}.txt" ,f"./record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/ELEMENT_{model}_{file_name}{tail1}.txt")
      else:
        shutil.copy(f"./record_{model}/CLUSTER/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{tail_distribute}.txt" ,f"./record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{tail1}.txt")
        shutil.copy(f"./record_{model}/ELEMENT/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/ELEMENT_{model}_{file_name}{tail_distribute}.txt" ,f"./record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/ELEMENT_{model}_{file_name}{tail1}.txt")
        shutil.rmtree(f"./record_{model}/ELEMENT/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}")
        shutil.rmtree(f"./record_{model}/CLUSTER/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}")
    
      iter1 = copy.deepcopy(iter)
      level1 = copy.deepcopy(level)
      count1 = copy.deepcopy(count)
      tile_grid1 = copy.deepcopy(tile_grid)
      mapping_info1 = copy.deepcopy(mapping_info)
      selected_list1= copy.deepcopy(selected_list)
      if placement == None:
        placement_list1 = None
      else:
        placement_list1 = copy.deepcopy(placement)
      copy_lock = 1
    
      cluster_f_select1 = open(f"./record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{tail1}.txt",'a')
      cluster_wr_select1 = csv.writer(cluster_f_select1)
      element_f_select1 = open(f"./record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/ELEMENT_{model}_{file_name}{tail1}.txt",'a')
      element_wr_select1 = csv.writer(element_f_select1)
      num_of_col1 = None

      if args.distribute == 1:
        iter1, count1, level1, tile_grid1,num_of_col1,addition_cluster,placement_list1= profile_for_select_distribute(i, tmp_dfg , Nbit ,chip1_buswidth, chip1_clk_freq, cluster_clk_freq, FPS ,CLUSTER1, iter1, cluster_wr_select1 ,element_wr_select1, shape1.get(conv_dense), level1, count1, chip_width, ht, tile_grid1,mapping_info1,placement_list1)
        if iter1 == None:
          return iter1,level1,count1,tile_grid1,mapping_info1,conv_dense,None,None,None,None,selected_list1,placement_list1
      else:
        iter1, count1, level1, tile_grid1,num_of_col1,addition_cluster= profile_for_select(i, tmp_dfg , Nbit ,chip1_buswidth, chip1_clk_freq, cluster_clk_freq, FPS ,CLUSTER1, iter1, cluster_wr_select1 ,element_wr_select1, shape1.get(conv_dense), level1, count1, chip_width, ht, tile_grid1,mapping_info1,placement_list1)
      
      cluster_f_select1.close()
      element_f_select1.close()

      booksim_latency1,booksim_energy1,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage = execute_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,model,tail1,cluster_meter,chip1_meter,chip1_period,period,cluster_buswidth,chip1_buswidth,file_name,depth, NUM_ROW_cluster, NUM_COL_cluster ,idx)

      inverse_key = tmp_dfg[str(i)][1]
      if tmp_dfg[str(i)][1] == '':
        inverse_key = tmp_dfg[str(i)][2]
      
      if inverse_key in inverse.keys():
        inverse[inverse_key][0][0] = inverse[inverse_key][0][0] + booksim_latency1 + compute_PPA1[conv_dense][0]
        inverse[inverse_key][0][1] = inverse[inverse_key][0][1] + booksim_energy1 + compute_PPA1[conv_dense][1]
      else:
        inverse[inverse_key] = [[booksim_latency1 + compute_PPA1[conv_dense][0], booksim_energy1 + compute_PPA1[conv_dense][1]],booksim_energy1] 
      
      before_node = i

      if dfg_key[-1] == i:
        total_latency1 = total_latency  + booksim_latency1 + compute_PPA1[conv_dense][0]
        total_energy1 = total_energy + booksim_energy1 + compute_PPA1[conv_dense][1]
        if depth == 0 :
          total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2))*chip_width + (cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
          total_area1 = total_area + cluster_booksim_area + (chip1_booksim_area+cluster_area)*addition_cluster
        else :
          total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
          total_area1 = total_area + (chip1_booksim_area+cluster_area)*addition_cluster
        selected_list1.append(ht)
        
     
        conv_dense=conv_dense+1
        return iter1,level1,count1,tile_grid1,mapping_info1,conv_dense,total_latency1,total_energy1,total_area1,total_leakage1,selected_list1, placement_list1

      conv_dense=conv_dense+1

    else:
      before_conv_result1 = inverse[i][0]
      key_loc_before_size=None
      before_node_list=[]
      for key_iter_for_fing_dest,p in enumerate(tmp_dfg.keys()):
          if tmp_dfg[p][1]==i or tmp_dfg[p][2]==i:
            key_loc_before_size = tmp_dfg[p][4]
            before_node_list.append(int(p))

      injection1 = injection_rate(key_loc_before_size/num_of_col1, Nbit, FPS,  chip1_buswidth, cluster_clk_freq)
      cluster_f_select1 = open(f"./record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{tail1}.txt",'a')
      cluster_wr_select1 = csv.writer(cluster_f_select1)
      cluster_wr_select1.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{iter1[0]}-0","non_MAC",key_loc_before_size/num_of_col1,injection1,1,1])
      cluster_f_select1.close()
      
 
      booksim_latency_non_mac = 0
      booksim_energy_non_mac = 0

      booksim_latency_non_mac,booksim_energy_non_mac,cluster_booksim_area_non_mac,chip1_booksim_area_non_mac,cluster_booksim_leakage_non_mac,chip1_booksim_leakage_non_mac  = execute_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,model,tail1,cluster_meter,chip1_meter,chip1_period,period,cluster_buswidth,chip1_buswidth,file_name,depth,NUM_ROW_cluster, NUM_COL_cluster, idx)

      total_latency1 = total_latency + booksim_latency_non_mac + before_conv_result1[0]
      total_energy1 = total_energy + booksim_energy_non_mac + before_conv_result1[1]
      if depth == 0 :
        total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2))*chip_width + (cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
        total_area1 = total_area + cluster_booksim_area + (chip1_booksim_area+cluster_area)*addition_cluster
      else :
        total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
        total_area1 = total_area + (chip1_booksim_area+cluster_area)*addition_cluster
      
      selected_list1.append(ht)
      
      leak_energy = (cluster_booksim_leakage/(chip_width**2)*addition_cluster) * (booksim_latency_non_mac + before_conv_result1[0]) *1000

      return iter1,level1,count1,tile_grid1,mapping_info1,conv_dense,total_latency1,total_energy1,total_area1,total_leakage1,selected_list1,placement_list1

def make_args_predict(ht,config,config_cluster,tmp_dfg,shape1,compute_PPA1,chip_width,FPS,iter,level,count,mapping_num_list,tile_grid,mapping_info,conv_dense,total_latency,total_energy,total_area,total_leakage,dfg_key,selected_list,file_name, depth, model, placement,idx,total_similarity): #현재 total_similarity 빠져 있음
  NUM1_ROW = config[0]
  NUM1_COL = config[1]
  PE1 = config[2]
  Tile1 = config[3]
  ADC=config[4]
  Cellbit=config[5]

  NUM_ROW_cluster = config_cluster[0]
  NUM_COL_cluster = config_cluster[1]
  PE_cluster = config_cluster[2]
  Tile_cluster = config_cluster[3]
  ADC_cluster=config_cluster[4]
  Cellbit_cluster=config_cluster[5]
  
  CLUSTER1 = int((NUM_ROW_cluster*Tile_cluster)/(NUM1_ROW*Tile1))
  chip1_flit_cycle = flit_cycle(unitLatencyRep.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"), unitLatencyWire.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"),tile_width_meter.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"),clk_frequency.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"),minDist.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"))
  chip1_clk_freq = clk_frequency.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  
  cluster_flit_cycle = flit_cycle(unitLatencyRep.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}"), unitLatencyWire.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}"),tile_width_meter.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}"),clk_frequency.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}"),minDist.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}"))
  cluster_meter = max(tile_width_meter.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")*CLUSTER1, tile_width_meter.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}"))
  cluster_clk_freq = clk_frequency.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}")
  
  chip1_meter = tile_width_meter.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  chip1_buswidth = busWidth.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  cluster_buswidth = chip1_buswidth
  chip1_period = clk_period.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  period = clk_period.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}")

  linear_model = predict_model_dict[f"{model},{ADC},{Cellbit}"][0]
  polynomial_features = predict_model_dict[f"{model},{ADC},{Cellbit}"][1]
  loaded_scaler_energy = predict_model_dict[f"{model},{ADC},{Cellbit}"][2]
  scaler_x_scaler_energy = predict_model_dict[f"{model},{ADC},{Cellbit}"][3]
  scaler_y_scaler_energy = predict_model_dict[f"{model},{ADC},{Cellbit}"][4]

  chip1_area = (chip1_meter**2)*1e12
  chip1_leakage =0 
  if f"layer{conv_dense}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}" in leakage_POWER:
    chip1_leakage = leakage_POWER.get(f"layer{conv_dense}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")*1e-6
  chip1_booksim_area  = 0
  cluster_booksim_area =0
  chip1_booksim_leakage  = 0
  cluster_booksim_leakage = 0
  cluster_area = (cluster_meter**2)*1e12
  tail_select=""
  for selected in selected_list:
    tail_select = tail_select + str(selected)
  
  tail_distribute = tail_select

  Nbit=8
  node_col={}
  inverse = {}
  tail1=f"{tail_select}{ht}"

  try:
    addition_cluster = 0
    for i in dfg_key:
      # print(f"---------------------{i}---------------------")
      # print(tmp_dfg[str(i)][0])  
      if (tmp_dfg[str(i)][0]=='nn.conv2d' or tmp_dfg[str(i)][0]=='nn.dense'):
        os.makedirs(f"./record_GA_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}", exist_ok=True)
        os.makedirs(f"./record_GA_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}", exist_ok=True)
        os.makedirs(f"./record_GA_{model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}", exist_ok=True)
        if depth == 0 :
          shutil.copy(f"./record_GA_{model}/CLUSTER/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{model}_{file_name}{tail_distribute}.txt" ,f"./record_GA_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{tail1}.txt")
          shutil.copy(f"./record_GA_{model}/ELEMENT/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{model}_{file_name}{tail_distribute}.txt" ,f"./record_GA_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/ELEMENT_{model}_{file_name}{tail1}.txt")
    
        else:
          shutil.copy(f"./record_GA_{model}/CLUSTER/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{tail_distribute}.txt" ,f"./record_GA_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{tail1}.txt")
          shutil.copy(f"./record_GA_{model}/ELEMENT/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/ELEMENT_{model}_{file_name}{tail_distribute}.txt" ,f"./record_GA_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/ELEMENT_{model}_{file_name}{tail1}.txt")
          shutil.rmtree(f"./record_GA_{model}/ELEMENT/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}")
          shutil.rmtree(f"./record_GA_{model}/CLUSTER/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}")
        
        iter1 = copy.deepcopy(iter)
        level1 = copy.deepcopy(level)
        count1 = copy.deepcopy(count)
        mapping_num_list1 = copy.deepcopy(mapping_num_list)
        tile_grid1 = copy.deepcopy(tile_grid)
        mapping_info1 = copy.deepcopy(mapping_info)
        selected_list1= copy.deepcopy(selected_list)
        if placement == None:
          placement_list1 = None
        else:
          placement_list1 = copy.deepcopy(placement)
        copy_lock = 1

        cluster_f_select1 = open(f"./record_GA_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{tail1}.txt",'a')
        cluster_wr_select1 = csv.writer(cluster_f_select1)
        element_f_select1 = open(f"./record_GA_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/ELEMENT_{model}_{file_name}{tail1}.txt",'a')
        element_wr_select1 = csv.writer(element_f_select1)
        num_of_col1 = None
        booksim_energy_cluster = 0
        booksim_latency_cluster = 0

        if args.distribute == 1:
          iter1, count1, level1, tile_grid1,num_of_col1,addition_cluster, placement_list1, list_for_predict, mapping_num_list1= profile_for_select_distribute_predict(i, tmp_dfg , Nbit ,chip1_buswidth, chip1_clk_freq, cluster_clk_freq, FPS ,CLUSTER1, iter1, cluster_wr_select1 ,element_wr_select1, shape1.get(conv_dense), level1, count1, chip_width, ht, tile_grid1,mapping_info1, placement_list1, mapping_num_list1, model)
          if iter1 == None:
            return iter1,level1,count1,mapping_num_list,tile_grid1,mapping_info1,conv_dense,None,None,None,None,None,None,None
        else:
          iter1, count1, mapping_num_list1 ,level1, tile_grid1,num_of_col1,addition_cluster= profile_for_select(i, tmp_dfg , Nbit ,chip1_buswidth, chip1_clk_freq, cluster_clk_freq, FPS ,CLUSTER1, iter1, cluster_wr_select1 ,element_wr_select1, shape1.get(conv_dense), level1, count1, mapping_num_list1,chip_width, ht, tile_grid1,mapping_info1)
        
        cluster_f_select1.close()
        element_f_select1.close()
        # print(f'--------------layer{conv_dense}-----------------')
        # print(f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}')
        # print("compute latency", compute_PPA1[conv_dense])
      
        if depth==0: #or depth ==1:
          booksim_latency1,booksim_energy1,booksim_latency_cluster,booksim_energy_cluster ,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage = predict_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,model,tail1,cluster_meter,chip1_meter,list_for_predict,1)
          Booksim_Leakage_Area[f"{model}_{idx}"] = [cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage]
      
        else:
          booksim_latency1,booksim_energy1,booksim_latency_cluster,booksim_energy_cluster = predict_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,model,tail1,cluster_meter,chip1_meter,list_for_predict,0)
          cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage =  Booksim_Leakage_Area[f"{model}_{idx}"]
        
        inverse_key = tmp_dfg[str(i)][1]
        if tmp_dfg[str(i)][1] == '':
          inverse_key = tmp_dfg[str(i)][2]
        
        if inverse_key in inverse.keys():
          inverse[inverse_key][0][0][0] = inverse[inverse_key][0][0][0] + booksim_latency1 
          inverse[inverse_key][0][0][1] = inverse[inverse_key][0][0][1] + compute_PPA1[conv_dense][0]
          inverse[inverse_key][0][1] = inverse[inverse_key][0][1] + booksim_energy1 + compute_PPA1[conv_dense][1]
        else:
          inverse[inverse_key] = [[[booksim_latency1 , compute_PPA1[conv_dense][0] ,booksim_latency_cluster], [booksim_energy1 , compute_PPA1[conv_dense][1], booksim_energy_cluster]]] 
        
        before_node = i
      
        if dfg_key[-1] == i:
          poly = polynomial_features.transform([[booksim_latency1, compute_PPA1[conv_dense][0]]])
          scale_latency = linear_model.predict(poly)
          
          booksim_energy = booksim_energy1 * (scale_latency[0][0]-compute_PPA1[conv_dense][0]) * 1000

          mlp_energy = scaler_x_scaler_energy.transform([[booksim_energy, compute_PPA1[conv_dense][1]]])
          scale_energy = loaded_scaler_energy.predict(mlp_energy)
          scale_energy = scaler_y_scaler_energy.inverse_transform(scale_energy.reshape(-1,1))
          
          total_latency1 = total_latency  + scale_latency[0][0]
          total_energy1 = total_energy + scale_energy[0][0]
          if depth == 0 :
            total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2))*chip_width + (cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
            total_area1 = total_area + cluster_booksim_area + (chip1_booksim_area+cluster_area)*addition_cluster

          else :
            total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
            total_area1 = total_area + (chip1_booksim_area+cluster_area)*addition_cluster
          selected_list1.append(ht)
          
      
          conv_dense=conv_dense+1
    
          if args.search_accuracy == 1:
            total_similarity1 = total_similarity + (Similarity_for_accuracy[f"{model},{ADC},{Cellbit}"][f"{NUM1_ROW},{NUM1_COL}"][conv_dense-1]* math.pow(hessian_list[f"{model}"][conv_dense-1],2))
            return iter1,level1,count1,mapping_num_list1,tile_grid1,mapping_info1,conv_dense,float(total_latency1),float(total_energy1),total_area1,total_leakage1,selected_list1,placement_list1,total_similarity1
          
          else:
            return iter1,level1,count1,mapping_num_list1,tile_grid1,mapping_info1,conv_dense,float(total_latency1),float(total_energy1),total_area1,total_leakage1,selected_list1,placement_list1,total_similarity

        conv_dense=conv_dense+1

      else:
        before_conv_result1 = inverse[i][0]
     
        key_loc_before_size=None
        before_node_list=[]
        for key_iter_for_fing_dest,p in enumerate(tmp_dfg.keys()):
            if tmp_dfg[p][1]==i or tmp_dfg[p][2]==i:
              key_loc_before_size = tmp_dfg[p][4]
              before_node_list.append(int(p))

        injection1 = injection_rate(key_loc_before_size/num_of_col1, Nbit, FPS,  chip1_buswidth, cluster_clk_freq)
        cluster_f_select1 = open(f"./record_GA_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/{idx}/CLUSTER_{model}_{file_name}{tail1}.txt",'a')
        cluster_wr_select1 = csv.writer(cluster_f_select1)
        cluster_wr_select1.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{iter1[0]}-0","non_MAC",key_loc_before_size/num_of_col1,injection1,1,1])
        cluster_f_select1.close()
        mapping_num_list1.append(1)
        booksim_latency_non_mac = 0
        booksim_energy_non_mac = 0
        
        st1 = node[model][i][1]
        st2 = node[model][i][2]
        
        key_position1 = node_keys_list[model].index(st1) if st1 in node_keys_list[model] else None
        key_position2 = node_keys_list[model].index(st2) if st2 in node_keys_list[model] else None

        input_cluster = 0
        if key_position1 is not None:
          input_cluster += mapping_num_list1[key_position1]
        if key_position2 is not None:
          input_cluster += mapping_num_list1[key_position2]

        list_for_predict={}
        list_for_predict["injection_cluster"] = injection1
        list_for_predict["injection_element_send"] = 0
        list_for_predict["injection_element_receive"] = 0
        
        list_for_predict["activation_size_cluster"] = key_loc_before_size/num_of_col1
        list_for_predict["activation_size_element_send"] = 0
        list_for_predict["activation_size_element_receive"] = 0

        list_for_predict["num_of_input_cluster"] =  input_cluster
        list_for_predict["num_of_input_element_send"] = 1
        list_for_predict["num_of_input_element_receive"] = 1

        list_for_predict["num_of_dest_cluster"] = mapping_num_list1[-1]
        list_for_predict["num_of_dest_element_send"] = 1
        list_for_predict["num_of_dest_element_receive"] = 1

        booksim_latency_non_mac,booksim_power_non_mac, booksim_latency_non_mac_cluster,booksim_energy_non_mac_cluster = predict_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,model,tail1,cluster_meter,chip1_meter, list_for_predict,0)
      
      
        # Open the file in read-binary mode and load the model
        if depth == 0 :
          before_conv_result1[0][0] += booksim_latency_non_mac
          total_latency1 = total_latency + before_conv_result1[0][0]+ before_conv_result1[0][1]
          
          before_conv_result1[1][0] += booksim_energy_non_mac
          total_energy1 = total_energy + before_conv_result1[1][0]+ before_conv_result1[1][1]
        
        else: 
          before_conv_result1[0][0] += booksim_latency_non_mac
          poly = polynomial_features.transform([[before_conv_result1[0][0],before_conv_result1[0][1]]])
          scale_latency = linear_model.predict(poly)

          if (args.heterogeneity > 1):
            poly_cluster = polynomial_features.transform([[before_conv_result1[0][2],before_conv_result1[0][1]]])
            scale_latency_cluster = linear_model.predict(poly_cluster)
          
          before_conv_result1[1][0] +=booksim_power_non_mac      
          mlp_energy = scaler_x_scaler_energy.transform([[before_conv_result1[1][0]*(scale_latency[0][0]-before_conv_result1[0][1])*1000, before_conv_result1[1][1]]])
          scale_energy = loaded_scaler_energy.predict(mlp_energy)
          scale_energy = scaler_y_scaler_energy.inverse_transform(scale_energy.reshape(-1,1))
          
          if (args.heterogeneity > 1):   
            mlp_energy_cluster = scaler_x_scaler_energy.transform([[before_conv_result1[1][2]*(scale_latency_cluster[0][0]-before_conv_result1[0][1])*1000, before_conv_result1[1][1]]])
            scale_energy_cluster = loaded_scaler_energy.predict(mlp_energy_cluster)
            scale_energy_cluster = scaler_y_scaler_energy.inverse_transform(scale_energy_cluster.reshape(-1,1))
          
          if (args.heterogeneity > 1): 
            total_latency1 = total_latency + scale_latency[0][0] + (scale_latency_cluster[0][0]-before_conv_result1[0][1])
            total_energy1 = total_energy + scale_energy[0][0] + (scale_energy_cluster[0][0]-before_conv_result1[1][1])
          else:
            total_latency1 = total_latency + scale_latency[0][0]
            total_energy1 = total_energy + scale_energy[0][0] 
        
        if depth == 0 :
          total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2))*chip_width + (cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
          total_area1 = total_area + cluster_booksim_area + (chip1_booksim_area+cluster_area)*addition_cluster
        else :
          total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
          total_area1 = total_area + (chip1_booksim_area+cluster_area)*addition_cluster
        
        
        selected_list1.append(ht)
        if args.search_accuracy == 1:
          total_similarity1 = total_similarity + (Similarity_for_accuracy[f"{model},{ADC},{Cellbit}"][f"{NUM1_ROW},{NUM1_COL}"][conv_dense-1]* math.pow(hessian_list[f"{model}"][conv_dense-1],2))
          return iter1,level1,count1,mapping_num_list1,tile_grid1,mapping_info1,conv_dense,float(total_latency1),float(total_energy1),total_area1,total_leakage1,selected_list1,placement_list1,total_similarity1
        
        else:
          return iter1,level1,count1,mapping_num_list1,tile_grid1,mapping_info1,conv_dense,float(total_latency1),float(total_energy1),total_area1,total_leakage1,selected_list1,placement_list1,total_similarity
  except Exception as e:
    error_info = traceback.format_exc()
    print(error_info)
    print("make_args_predict")

  

def execute_simulation(CONFIG,model,placement,chip_width,idx):
  config = CONFIG[0]
  tile_select = CONFIG[1]  
  mapping_num_list_local = []
  depth = 0
  if placement == None and chip_width == 0:
    placement_local = None
    iter_local = iter_list[f'{model}_{config}'].copy()
    level_local = level_list[f'{model}_{config}'].copy()
    count_local = count_list[f'{model}_{config}'].copy()
    tile_grid_local  = tile_grid_list[f'{model}_{config}'].copy()
    mapping_info_local  = mapping_info_list[f'{model}_{config}'].copy()
    conv_dense_local  = conv_dense_list[f'{model}_{config}']
    total_latency_local  = total_latency_list[f'{model}_{config}']
    total_energy_local  = total_energy_list[f'{model}_{config}']
    total_leakage_local  = total_leakage_list[f'{model}_{config}']
    total_area_local  = total_area_list[f'{model}_{config}']
    total_similarity_local = total_similarity_list[f'{model}_{config}']
    all_selected_local  = all_selected_list[f'{model}_{config}'].copy()
    
    chip_width = chip_width_list[f'{model}_{config}']
  
  else:
    placement_local = placement.copy()
    iter_local = iter_list[f'{model}_{config}'].copy()
    level_local = level_list[f'{model}_{config}'].copy()
    count_local = count_list[f'{model}_{config}'].copy()
    tile_grid_local  = tile_grid_list[f'{model}_{config}'].copy()
    mapping_info_local  = mapping_info_list[f'{model}_{config}'].copy()
    conv_dense_local  = conv_dense_list[f'{model}_{config}']
    total_latency_local  = total_latency_list[f'{model}_{config}']
    total_energy_local  = total_energy_list[f'{model}_{config}']
    total_leakage_local  = total_leakage_list[f'{model}_{config}']
    total_area_local  = total_area_list[f'{model}_{config}']
    total_similarity_local = total_similarity_list[f'{model}_{config}']
    all_selected_local  = all_selected_list[f'{model}_{config}'].copy()

  try:
    for exec_set in execute_set[model]:
      if not os.path.isdir(f"{path}/CLUSTER/{args.heterogeneity}/{depth}"):
          os.makedirs(f"{path}/CLUSTER/{args.heterogeneity}/{depth}")
      if not os.path.isdir(f"{path}/ELEMENT/{args.heterogeneity}/{depth}"):
          os.makedirs(f"{path}/ELEMENT/{args.heterogeneity}/{depth}")
      if not os.path.isdir(f"{path}/BOOKSIM/{args.heterogeneity}/{depth}"):
          os.makedirs(f"{path}/BOOKSIM/{args.heterogeneity}/{depth}")

      config = CONFIG[0]
      config_origin = find_origin_config[f'{model}_{config}']
      file_name = make_file_name(config_origin)
      ht = int(tile_select[depth])

      if cnt < args.generation and nostop:
        iter_local,level_local,count_local,mapping_num_list_local,tile_grid_local,mapping_info_local,conv_dense_local,total_latency_local,total_energy_local,total_area_local,total_leakage_local,all_selected_local,placement_local,total_similarity_local = make_args_predict(ht,config_origin[ht],config_origin[0], dfg[model], globals()['shape{}'.format(ht)][f'{model}_{config_origin}'] , globals()['compute_PPA{}_list'.format(ht)][f'{model}_{config_origin}'], chip_width, FPS_list[f'{model}_{config}'], iter_local, level_local, count_local, mapping_num_list_local, tile_grid_local, mapping_info_local, conv_dense_local, total_latency_local, total_energy_local,total_area_local,total_leakage_local,exec_set, all_selected_local, file_name, depth, model, placement_local,idx,total_similarity_local)
        if iter_local == None:
          return None,None,None,None
      else:
        iter_local,level_local,count_local,tile_grid_local,mapping_info_local,conv_dense_local,total_latency_local,total_energy_local,total_area_local,total_leakage_local,all_selected_local,placement_local = make_args_simulator(ht,config_origin[ht],config_origin[0], dfg[model], globals()['shape{}'.format(ht)][f'{model}_{config_origin}'] , globals()['compute_PPA{}_list'.format(ht)][f'{model}_{config_origin}'], chip_width, FPS_list[f'{model}_{config}'], iter_local, level_local, count_local, tile_grid_local, mapping_info_local, conv_dense_local, total_latency_local, total_energy_local,total_area_local,total_leakage_local,exec_set, all_selected_local, file_name, depth, model, placement_local,idx)
        total_similarity_local = 0
        if iter_local == None:
          return None,None,None,None
      depth+=1
  except Exception as e:
    print(e, depth)

  try:
    (total_energy_local/total_latency_local)
  except Exception as e:
    print(total_latency_local, total_energy_local, total_area_local)
  
  if placement_local == None:
    return total_latency_local, (total_energy_local/total_latency_local)+total_leakage_local,total_area_local, total_similarity_local, mapping_info_local  ## latency, power, area
  else:
    return total_latency_local, (total_energy_local/total_latency_local)+total_leakage_local,total_area_local, total_similarity_local

#args comes from tile_mapping_set with multiple sets per config
def internal_parallel(arg_list):
  comb_set =arg_list[1]
  idx = arg_list[0]
  tmp_ =[]

  try:
    first = [(t,model_list[i]) for i,t in enumerate(comb_set) if len(t) == 3]
    latency, power, area, accuracy, mapping_info = execute_simulation(first[0][0], first[0][1], None, 0,idx)
    mapping_info = str(mapping_info)
    where_tile = re.findall(r'\d+-\d+', mapping_info)
    which_tile = re.findall(r"(\d+), array",mapping_info)
    mapping_tmp = []
    arg =[]
    for h in range(args.heterogeneity):
        mapping_tmp.append([])
    for i in range(len(where_tile)):
      for j in range(args.heterogeneity):
        if which_tile[i] == str(j):
          mapping_tmp[j].append(where_tile[i])
    for h in range(args.heterogeneity):
        mapping_tmp[h] = ','.join(mapping_tmp[h])
    arg = "["
    for h in range(args.heterogeneity):
        arg += f"[{mapping_tmp[h]}]"
    arg += "]"

    list_strings = arg.strip("[]").split("][")
    placement = [group.split(",") for group in list_strings]
    if cnt < args.generation and nostop:
      if args.search_accuracy == 1:
        first_reslut = (latency, power, area, accuracy)
      else:
        first_reslut = (latency, power, area)
    else:
      if args.search_accuracy == 1:
        first_reslut = (latency, power, area, accuracy)
      else:
        first_reslut = (latency, power, area)
    fixed_chip_width = chip_width_list[f'{first[0][1]}_{first[0][0][0]}']

    for i,comb in enumerate(comb_set):
      if len(comb) == 2:
        latency, power, area, accuracy = execute_simulation(comb,model_list[i],placement,fixed_chip_width,idx) 
        if cnt < args.generation and nostop:
          if args.search_accuracy == 1:
            tmp_.append((latency, power, area, accuracy))
          else:
            tmp_.append((latency, power, area))
        else:
          if args.search_accuracy == 1:
            tmp_.append((latency, power, area, accuracy))
          else:
            tmp_.append((latency, power, area))
      else:
        tmp_.append(first_reslut)

  except Exception as e:
    print(e, "internal_parallel")
    print(len(which_tile), len(where_tile))
    print(arg_list)
    error_info = traceback.format_exc()
    print(error_info)
  if args.search_accuracy == 1:
    if (None,None,None,None) not in tmp_:
      results[str(comb_set)] = tmp_
  else:
    if (None,None,None) not in tmp_:
      results[str(comb_set)] = tmp_
###main
model_list = args.models.split(',')
numbers = re.findall(r'\d+', args.weights)
numbers = list(map(int, numbers))
model_weight = [numbers[i:i+3] for i in range(0, len(numbers), 3)]
#tile_selct num table
tile_num_table = [[] for i in range(len(model_list))]
config_zip =[]
for i in range(len(model_list)):
  path =f"{navcim_dir}/Inference_pytorch/shape/shape_{model_list[i]}_*"
  files = glob.glob(path, recursive=True)
  files = [f for f in files if os.path.isfile(f)]
  files = sorted(files,key=sort_key)

  if i > 0: 
    for file in files:
      with open(file, 'r') as f:
        v = f.readlines()
        tile_tmp = []
        for j in v:
            tile_tmp.append(int(j.split(',')[-1]))
        tile_num_table[i].append(tile_tmp)
  else:
     for file in files:
        numbers = re.findall(r'ADC:(\d+)_Cellbit:(\d+)_SA_row:(\d+)_SA_col:(\d+)_PE:(\d+)_TL:(\d+)', file)
        result = [int(num) for num in numbers[0]] if numbers else []
        result = result[2:] + result[:2]
        config_zip.append(result)
        
        with open(file, 'r') as f:
          v = f.readlines()
          tile_tmp = []
          for j in v:
            tile_tmp.append(int(j.split(',')[-1]))
          tile_num_table[i].append(tile_tmp)



all_data = {}
config_order ={}
best_result = {}

LATENCY={}
POWER={}
AREA={}
inter_connect_LATENCY={}
inter_connect_POWER={}
inter_connect_AREA={}
hessian_list = {}
Similarity_for_accuracy = {}
parameter_crosssim = {}
ntest = 100
ntest_batch = 100

leakage_POWER={}
tile_width_meter={}
clk_frequency={}
clk_period={}
unitLatencyRep={}
unitLatencyWire={}
minDist={}
busWidth={}
node={}
dfg={}
execute_set={}

chip_width_list=Manager().dict()
FPS_list=Manager().dict()
iter_list=Manager().dict()
level_list=Manager().dict()
count_list=Manager().dict()
tile_grid_list=Manager().dict()
mapping_info_list=Manager().dict()
conv_dense_list=Manager().dict()
total_latency_list=Manager().dict()
total_energy_list=Manager().dict()
total_area_list=Manager().dict()
all_selected_list=Manager().dict()
total_leakage_list=Manager().dict()
find_origin_config = Manager().dict()
node_keys_list = Manager().dict()
Booksim_Leakage_Area=Manager().dict()
total_similarity_list=Manager().dict()

for i in range(args.heterogeneity):
    globals()['shape{}'.format(i)] = Manager().dict()
    globals()['compute_PPA{}_list'.format(i)] = Manager().dict()
lock = Manager().Lock() 
results = Manager().dict()

for i,model in enumerate(model_list):
  if args.search_accuracy == 1:
    file_path = f"{navcim_dir}/Inference_pytorch/search_result/{args.models}_multi_model_step3/final_result_{model}_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{model_weight[i][0]},{model_weight[i][1]},{model_weight[i][2]},{args.accuracy}]_{args.heterogeneity}_cka.txt"
  else:
    file_path = f"{navcim_dir}/Inference_pytorch/search_result/{args.models}_multi_model_step3/final_result_{model}_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{model_weight[i][0]},{model_weight[i][1]},{model_weight[i][2]}]_{args.heterogeneity}_pareto.txt"
  model_data = read_and_sort_file(file_path)
  model_data = sorted(model_data, key=lambda x: tuple(x[0]))
  a = len(model_data)
  print(a)

  all_data[model] = model_data
  config_order[model] = [i[0] for i in model_data]
##
# filtering config_orders
common_config_identifiers = find_common_configs(config_order)
filter_configs(config_order, common_config_identifiers)
for model in model_list:
  filtered_data = [data for data in all_data[model] if data[0] in config_order[model]]
  all_data[model] = filtered_data
##
print("Config Num: ", len(filtered_data))

adc_set = []
cellbit_set = []
tile_name = [] 
for cf in config_order[model]:
    for c in cf:
      if c not in tile_name:
        tile_name.append(c)
        if c[4] not in adc_set:
          adc_set.append(c[4])
        if c[5] not in cellbit_set:
          cellbit_set.append(c[5])

predict_model_dict={}

for model in model_list:
  All_layer = []
 
  for adc in adc_set:
    for cellbit in cellbit_set:
      #path pkl

      loaded_model_filename = f'{navcim_dir}/Inference_pytorch/predict_model/{model}/polynomial_regression_{model}_latency_{adc}_{cellbit}final.pkl'
      loaded_poly_filename = f'{navcim_dir}/Inference_pytorch/predict_model/{model}/polynomial_features_{model}_latency_{adc}_{cellbit}final.pkl'

      with open(loaded_model_filename, 'rb') as file:
          linear_model = joblib.load(file)

      with open(loaded_poly_filename, 'rb') as file:
          polynomial_features = joblib.load(file) 

      loaded_model_filename_energy = f'{navcim_dir}/Inference_pytorch/predict_model/{model}/mlp_{model}_energy_{adc}_{cellbit}_scale_final.pkl'
      loaded_scaler_filename_energy = f'{navcim_dir}/Inference_pytorch/predict_model/{model}/mlp_scaler_x_{model}_energy_{adc}_{cellbit}final.pkl'
      loaded_scaler_y_filename_energy = f'{navcim_dir}/Inference_pytorch/predict_model/{model}/mlp_scaler_y_{model}_energy_{adc}_{cellbit}final.pkl'

      with open(loaded_model_filename_energy, 'rb') as file:
          loaded_scaler_energy = pickle.load(file)

      scaler_x_scaler_param_energy = joblib.load(loaded_scaler_filename_energy)

      scaler_x_scaler_energy = StandardScaler()
      scaler_x_scaler_energy.mean_ = scaler_x_scaler_param_energy['mean']
      scaler_x_scaler_energy.scale_ = scaler_x_scaler_param_energy['scale']

      scaler_y_scaler_param_energy = joblib.load(loaded_scaler_y_filename_energy)

      scaler_y_scaler_energy = StandardScaler()
      scaler_y_scaler_energy.mean_ = scaler_y_scaler_param_energy['mean']
      scaler_y_scaler_energy.scale_ = scaler_y_scaler_param_energy['scale']
      
      predict_model_dict[f"{model},{adc},{cellbit}"] = [linear_model,polynomial_features ,loaded_scaler_energy,scaler_x_scaler_energy,scaler_y_scaler_energy]
  
  if args.search_accuracy == 1:
  #raise error
    name_model = model
    for adc in adc_set:
      for cellbit in cellbit_set:
        df = pd.read_csv(f'{navcim_dir}/cross-sim/applications/dnn/inference/{name_model}_cka_ADC:{adc}_CellBit{cellbit}_list.csv')
        Similarity_for_accuracy_tmp = {col: df[col].tolist() for col in df.columns}
        Similarity_for_accuracy[f"{model},{adc},{cellbit}"] = Similarity_for_accuracy_tmp

        layerParams, sizes, onnx_model, quantize, adc_ranges_cadidate, index_row_col, dac_ranges, positiveInputsOnly, xy_pars, weight_for_quantized, config =  deepcopy(run_inference_for_search.init(name_model,ntest, ntest_batch,cellbit,adc))
        parameter_crosssim[f"{model},{adc},{cellbit}"] = [layerParams, sizes, onnx_model, quantize, adc_ranges_cadidate, index_row_col, dac_ranges, positiveInputsOnly, xy_pars, weight_for_quantized,config]
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
    print_configuration_message(config, f"NavCim_log/{args.models}/accuracy_true/Tile_{tile_set}/PE_{pe_set}/SA_{sa_set}/ADC_{adc_set}/CellBit_{cellbit_set}/heterogeneity_{args.heterogeneity}/{args.date}/CrossSim_configuration.txt")
  
    with open(f'{navcim_dir}/cross-sim/applications/dnn/inference/{name_model}_hessian_list.txt', 'r') as file:
      content = file.read() 
    hessian_list[f"{model}"] = [float(number) for number in content.split(', ')]
  
  
  
  for config in tile_name:
    NUM_ROW = config[0]
    NUM_COL = config[1]
    PE = config[2]
    Tile = config[3]
    ADC = config[4]
    Cellbit = config[5]
    fname = f"./summary/summary_{model}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}.txt"
    # print(fname)
    with open(fname) as f:
      lines = f.readlines()
      for i, l in enumerate(lines):
        if l.find("readLatency is:")!= -1 and l.find("layer")!= -1:
          layername = l.split("\'")[0]
          l=l.split("\'")[1]
          latency = l.split(":")[1]
          latency = float(latency[:-3])
          LATENCY[f'{model}_{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=latency
          if layername not in All_layer:
            All_layer.append(layername)
        elif l.find("s readDynamicEnergy is:")!= -1 and l.find("layer")!= -1:
          layername = l.split("\'")[0]
          l=l.split("\'")[1]
          readDynamicEnergy = l.split(":")[1]
          readDynamicEnergy = float(readDynamicEnergy[:-3])
          POWER[f'{model}_{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=readDynamicEnergy
        elif l.find("leakagePower is:")!= -1 and l.find("layer")!= -1:
          layername = l.split("\'")[0]
          l=l.split("\'")[1]
          leakagePower = l.split(":")[1]
          leakagePower = float(leakagePower[:-3])
          leakage_POWER[f'{model}_{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=leakagePower
        elif l.find("Tile Area is:")!= -1 and l.find("layer")!= -1:
          layername = l.split("\'")[0]
          l=l.split("\'")[1]
          cimarea = l.split(":")[1]
          cimarea = float(cimarea[:-5])
          AREA[f'{model}_{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=cimarea
        elif l.find("H-tree Area is")!= -1 and l.find("layer")!= -1:
          layername = l.split("\'")[0]
          l=l.split("\'")[1]
          inter_area = l.split(":")[1]
          inter_area = float(inter_area[:-5])
          inter_connect_AREA[f'{model}_{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=inter_area
        elif l.find("H-tree Latency is")!= -1 and l.find("layer")!= -1:
          layername = l.split("\'")[0]
          l=l.split("\'")[1]
          inter_latency = l.split(":")[1]
          inter_latency = float(inter_latency[:-3])
          inter_connect_LATENCY[f'{model}_{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=inter_latency
        elif l.find("H-tree Energy is")!= -1 and l.find("layer")!= -1:
          layername = l.split("\'")[0]
          l=l.split("\'")[1]
          inter_energy = l.split(":")[1]
          inter_energy = float(inter_energy[:-3])
          inter_connect_POWER[f'{model}_{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=inter_energy
        elif l.find("unitLatencyRep")!= -1:
          LatencyRep = l.split(":")[1].split("unitLatencyWire is")[0].strip()
          LatencyWire = l.split(":")[1].split("unitLatencyWire is")[1].strip()
          unitLatencyRep[f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= float(LatencyRep)
          unitLatencyWire[f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= float(LatencyWire)
        elif l.find("Tilewidth")!= -1:
          Tilewidth = l.split(":")[1]
          Tilewidth = float(Tilewidth[:-2])
          tile_width_meter[f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= Tilewidth
        elif l.find("Chip clock period")!= -1:
          clock_period = l.split(":")[1]
          clock_period = float(clock_period[:-3])
          # if clock_period == 0:
          clock_period = (6.50252e-3)*20
          # clock_period = 0.1
          clock_freq = (1/clock_period)*1e+9
          clk_period[f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= clock_period
          clk_frequency[f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= clock_freq
        elif l.find("minDist")!= -1:
          Dist = l.split("minDist")[1]
          Dist = float(Dist[:-2])
          minDist[f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= Dist
        elif l.find("busWidth")!= -1:
          bus = float(l.split("busWidth")[1])
          busWidth[f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= 128


  x = torch.randn(args.batch_size,3,224,224)
  assert model in ['ResNet50','EfficientNetB0','MobileNetV2','SqueezeNet'], model
  if model == 'ResNet50':
      modelCF = onnx.load("./network/resnet50.onnx")
  elif model == 'EfficientNetB0':
      modelCF = onnx.load("./network/efficientnet_b0.onnx")
  elif model == 'MobileNetV2':
      modelCF = onnx.load("./network/mobilenet_v2.onnx")
  elif model == 'SqueezeNet':
      modelCF = onnx.load("./network/squeezenet.onnx")
  else:
      raise ValueError("Unknown model type")

  graph, params = tvm.relay.frontend.from_onnx(modelCF, shape=None, dtype='float32', opset=None, freeze_params=True, convert_config=None)
  # graph, params= tvm.relay.optimize(graph, target='llvm', params=params)
  graph_ir = str(graph)
  parse_ir = graph_ir.split('\n')

  node_tmp={}
  dfg_tmp={}

  for i in parse_ir:
      name = ""
      type = ""
      input = ""
      input2 = ""
      kernel_size = ""
      stride = ""
      
      o = i.partition("/*")[0]
      j = o.partition("%")[2]
      k = j.partition("=")
      name = k[0]
      if "stride" in i:
          stride = i.partition("strides=")[2].partition(",")[0].partition("[")[2]    
      if "kernel_size" in i:
          kernel_size = i.partition("kernel_size=")[2].partition(",")[0].partition("[")[2]

      if "(" in k[2]:
          type = k[2].partition("(")[0]
          if "," in k[2]:
              if "[" in k[2]:
                  input = k[2].partition("(")[2].partition(",")[0].partition("%")[2]
              else:
                  input = k[2].partition("(")[2].partition(",")[0].partition("%")[2]
                  input2 = k[2].partition("(")[2].partition(",")[2].partition(")")[0].partition("%")[2]
          else:
              input = k[2].partition("%")[2].partition(")")[0]
      else:
          type =""
          input = k[2].partition("%")[2]

      activation = i.rpartition("/*")[2].partition("(")[2].partition(")")[0].split(",")
      if type.strip() == "nn.batch_norm":
          activation = i.rpartition("/*")[2].partition("(")[2].partition("(")[2].partition(")")[0].split(",")
      
      for h in range(len(activation)):
          activation[h] = activation[h].strip()
      if input != "":
          if "group" in i:
              node_tmp[name.strip()]=["depthwise",input.strip(),input2.strip(),activation,kernel_size,stride]
          else:
              node_tmp[name.strip()]=[type.strip(),input.strip(),input2.strip(),activation,kernel_size,stride]
  last_node=None

  for i in node_tmp.keys():
      if str(i) != '0':
          if node_tmp[str(i)][1] != '':
              if '.' in node_tmp[str(i)][1]:
                  node_tmp[str(i)][1] =  str(int(float(node_tmp[str(i)][1])))
          if node_tmp[str(i)][2] != '':
              if '.' in node_tmp[str(i)][2]:
                  node_tmp[str(i)][2] =  str(int(float(node_tmp[str(i)][2])))
  for i in node_tmp.keys():
    if node_tmp[str(i)][0]=='': 
      for j in node_tmp.keys():
        if node_tmp[str(j)][1]==str(i):
          node_tmp[str(j)][1] =  node_tmp[str(i)][1]
          node_tmp[str(j)][2] =  node_tmp[str(i)][2]
  for i in node_tmp.keys():
      last_node = i
      if node_tmp[str(i)][0]!='nn.dense' and node_tmp[str(i)][0]!='nn.conv2d' and node_tmp[str(i)][0]!='non-MAC':
          for j in node_tmp.keys():
              if node_tmp[str(j)][1]==str(i):
                  if node_tmp[str(j)][0]=="nn.conv2d" or node_tmp[str(j)][0]=="nn.dense" :
                      node_tmp[str(i)][0]="non-MAC"
                      break
                  else:
                      has_two=0
                  if node_tmp[str(i)][1] != '':
                      node_tmp[str(j)][1]=str(int(float(node_tmp[str(i)][1])))
                      has_two=1
                  if node_tmp[str(i)][2] != '':
                      if has_two == 1 and (model == "ResNet50" or model== "DenseNet40" or model=="RegNet_y"):
                          node_tmp[str(j)][2]=str(int(float(node_tmp[str(i)][2])))
                      else:
                          node_tmp[str(j)][1]=str(int(float(node_tmp[str(i)][2])))
              
              if node_tmp[str(j)][2]==str(i):
                  if node_tmp[str(j)][0]=="nn.conv2d" or node_tmp[str(j)][0]=="nn.dense" :
                      node_tmp[str(i)][0]="non-MAC"
                      break
                  else:
                      has_two=0
                  if node_tmp[str(i)][1] != '':
                      node_tmp[str(j)][2]=str(int(float(node_tmp[str(i)][1])))
                      has_two=1
                  if node_tmp[str(i)][2] != '':
                      if has_two == 1 and (model == "ResNet50" or model== "DenseNet40" or model=="RegNet_y"):
                          node_tmp[str(j)][1]=str(int(float(node_tmp[str(i)][2])))
                      else:
                          node_tmp[str(j)][2]=str(int(float(node_tmp[str(i)][2])))

  pop_list=[]
  for i in node_tmp.keys():
      if node_tmp[str(i)][0]!='nn.dense' and node_tmp[str(i)][0]!='nn.conv2d' and node_tmp[str(i)][0]!='non-MAC':
          pop_list.append(i)

  for i in pop_list:
      node_tmp.pop(str(i),None)
  
  non_MAC_list={}
  for i in node_tmp.keys():
    if node_tmp[str(i)][0]=='non-MAC':
      non_MAC_list[i]=node_tmp[str(i)][1:3]
  
  already_arrived = []
  for i in non_MAC_list.keys():
    if non_MAC_list[str(i)][0] != '':
      if non_MAC_list[str(i)][0] in non_MAC_list.keys():
        non_MAC_list[str(i)][0] =''
    
      if not non_MAC_list[str(i)][0] in already_arrived:
        already_arrived.append(non_MAC_list[str(i)][0])
      else:
        non_MAC_list[str(i)][0] =''
    
    if non_MAC_list[str(i)][1] !='':
      if non_MAC_list[str(i)][1] in non_MAC_list.keys():
        non_MAC_list[str(i)][1] = ''
      if not non_MAC_list[str(i)][1] in already_arrived:
        already_arrived.append(non_MAC_list[str(i)][1])
      else:
        non_MAC_list[str(i)][1] = ''

    if non_MAC_list[str(i)][0] != '':
      node_tmp[str(i)][1] = non_MAC_list[str(i)][0]
      node_tmp[str(i)][2] = non_MAC_list[str(i)][1]
    else:
      node_tmp[str(i)][1] = non_MAC_list[str(i)][1]
      node_tmp[str(i)][2] = non_MAC_list[str(i)][0]

  node_keys_list[model] = list(node.keys())
  for node_key in node_tmp.keys():
      key = node_key
      op= node_tmp[node_key][0]
      act= node_tmp[node_key][3]
      activation_size = None
      for act_ in act:
          if activation_size == None:
              activation_size = int(act_)
          else:
              activation_size = activation_size * int(act_)
      output1=''
      output2=''
      
      for tmp_key in node_tmp.keys():
          if node_tmp[tmp_key][1]==node_key:
              if output1=='':
                  output1=tmp_key
              else:
                  output2=tmp_key
          if node_tmp[tmp_key][2]==node_key:
              if output1=='':
                  output1=tmp_key
              else:
                  output2=tmp_key

      dfg_tmp[key]=[op,output1,output2,act,activation_size,1,node_tmp[node_key][4],node_tmp[node_key][5]]

  path = f"{navcim_dir}/Inference_pytorch/record_{model}"
  if not os.path.isdir(path):
      os.makedirs(path)
      os.makedirs(f"{path}/CLUSTER")
      os.makedirs(f"{path}/BOOKSIM")


  execute_set_tmp=[]
  tmp_set=[]

  for dfg_node in dfg_tmp.keys():
      if dfg_tmp[dfg_node][0] == 'nn.conv2d' or dfg_tmp[dfg_node][0] == 'nn.dense':
          if len(tmp_set) > 0:
              execute_set_tmp.append(tmp_set)
              tmp_set=[dfg_node]
          else:
              tmp_set=[dfg_node]
      else:
          tmp_set.append(dfg_node)
          execute_set_tmp.append(tmp_set)
          tmp_set=[]
  execute_set_tmp.append(tmp_set.copy())

  dfg[model] = dfg_tmp.copy()
  node[model] = node_tmp.copy()
  execute_set[model] = execute_set_tmp.copy()

  global_depth = -1
  
  for config in config_order[model]:
    if not os.path.isdir(f"{path}/CLUSTER/{global_depth}/{config[0][0]}/{config[0][1]}"):
      os.makedirs(f"{path}/CLUSTER/{global_depth}/{config[0][0]}/{config[0][1]}")
    if not os.path.isdir(f"{path}/ELEMENT/{global_depth}/{config[0][0]}/{config[0][1]}"):
      os.makedirs(f"{path}/ELEMENT/{global_depth}/{config[0][0]}/{config[0][1]}")
    if not os.path.isdir(f"{path}/BOOKSIM/{global_depth}/{config[0][0]}/{config[0][1]}"):
      os.makedirs(f"{path}/BOOKSIM/{global_depth}/{config[0][0]}/{config[0][1]}")

    shape_list, PPA_list ,chip_width_list[f'{model}_{config}'],FPS_list[f'{model}_{config}'],iter_list[f'{model}_{config}'],level_list[f'{model}_{config}'],count_list[f'{model}_{config}'],tile_grid_list[f'{model}_{config}'],mapping_info_list[f'{model}_{config}'],conv_dense_list[f'{model}_{config}'],total_latency_list[f'{model}_{config}'],total_energy_list[f'{model}_{config}'],total_area_list[f'{model}_{config}'],total_leakage_list[f'{model}_{config}'],all_selected_list[f'{model}_{config}'] = initialization(config, global_depth, dfg_tmp ,model)
    total_similarity_list[f'{model}_{config}'] = 0
    for i in range(args.heterogeneity):
      globals()['shape{}'.format(i)][f'{model}_{config}'] = shape_list[i]
      globals()['compute_PPA{}_list'.format(i)][f'{model}_{config}']= PPA_list[i]
    find_origin_config[f'{model}_{config}'] = config
  

print('GA START', args.generation)
nostop = True
cnt = 1
stop_count = 0
cand_all = [ [] for i in range(len(model_list))]
tile_mapping_set=[[] for i in range(len(all_data[model]))]
tile_mapping_set_tmp = Manager().list()

while cnt <= args.generation:
  print("generation",cnt)
  tmp_dic = {}
  tmp = []
  for model in model_list:
    tmp_ = [[] for i in range(len(all_data[model]))]
    if cnt > 1: 
      with Pool() as pool:
        cand = pool.map(run_ga_after_first_generation, cand_all[model_list.index(model)])
    else:
      with Pool() as pool:
        cand = pool.map(run_ga_first_generation, all_data[model])
    # print("cand", cand)
    for i in range(len(all_data[model])):
      if cand[i] != []:
        for j in range(args.population_size):
          tmp_[i].append((all_data[model][i][0],cand[i][j]))
      else:
        tmp_[i].append((all_data[model][i][0],cand[i]))
    tmp.append(tmp_)
  
  #view tile mappability
  again_gen = True
  again_gen_cnt = 1

  while again_gen:
    if again_gen_cnt > 15 :
      break
    print("Coverage codes")
 
    if again_gen_cnt == 1:
      with ProcessPoolExecutor() as executor:
        executor.map(check_is_map, zip(*tmp))
      ## tmp[0] == number of config
      print("num_set: ",len(tile_mapping_set_tmp))
      for comb_sub in tile_mapping_set_tmp:
        if comb_sub != []:
          for index, tmp_conf in enumerate(tmp[0]):
            try:
              if comb_sub[0][0][0] == tmp_conf[0][0]:
                tile_mapping_set[index].append(comb_sub)
                print(index == config_order[model_list[0]].index(comb_sub[0][0][0]))
                break
            except Exception as e:
              error_info = traceback.format_exc()
              print(error_info)          
    else:
      tile_mapping_set_tmp = Manager().list()
      print("tile set: ", tile_mapping_set_tmp)
      tmp_for_gen = regen_tmp.copy()

      with ProcessPoolExecutor() as executor:
        executor.map(check_is_map, zip(*tmp_for_gen))
     
      ## tmp[0] == number of config
      for comb_sublist in zip(tile_mapping_set_tmp, again_gen_index):
        if comb_sublist[0] != []:
          tile_mapping_set[comb_sublist[1]].append(comb_sublist[0])
        else:
          print(comb_sublist[0])
        
    # update the config count
    if [] not in tile_mapping_set:
      again_gen = False
    else:
      regen_tmp = []
      again_gen_index = [index for index, sub_list in enumerate(tile_mapping_set) if not sub_list]  ## 다시 해야할 config의 index 목록
      print(len(again_gen_index))
      
      regen_tmp_ = [[] for i in again_gen_index]
      for i,index in enumerate(again_gen_index):
        if tmp[0][index] != []:
          regen_tmp_[i] = tmp[0][index]
      regen_tmp.append(regen_tmp_)
      
      for m in range(1,len(model_list)):
        cand_0 = [[elem[1] for elem in tmp[m][index]] for index in again_gen_index]

        args.mutation_rate += 0.05 * again_gen_cnt
        try:
          with Pool() as pool:
            cand = pool.map(run_ga_after_first_generation, cand_0)
        except Exception as e:
          print(cand_0)
        args.mutation_rate -= 0.05 * again_gen_cnt
        
        regen_tmp_ = [[] for i in again_gen_index]
        for i,index in enumerate(again_gen_index):  
          if cand[i] != []:
            for j in range(args.population_size):
              regen_tmp_[i].append((all_data[model_list[m]][index][0],cand[i][j]))
        
        regen_tmp.append(regen_tmp_)
      again_gen_cnt += 1
      
       
  #compute PPA  
  parallel_list = []
  idx_cnt=0
  if cnt < args.generation and nostop:
    for args_list in tile_mapping_set:
      # print(args_list)
      if args_list!=[]:
        for comb_set in args_list[0]:
          parallel_list.append((idx_cnt, comb_set))
          idx_cnt+=1

    print("Start compute PPA")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
      executor.map(internal_parallel, parallel_list)
      
  else:
    keep_folder_name = f"./record_GA_{model}/CLUSTER/{args.heterogeneity}/-1"
    base_path = f"./record_GA_{model}/CLUSTER/{args.heterogeneity}"
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)  
        if os.path.isdir(item_path) and item != keep_folder_name:
            shutil.rmtree(item_path)
    
    keep_folder_name = f"./record_GA_{model}/BOOKSIM/{args.heterogeneity}/-1"
    base_path = f"./record_GA_{model}/BOOKSIM/{args.heterogeneity}"
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)  
        if os.path.isdir(item_path) and item != keep_folder_name:
            shutil.rmtree(item_path)  
    
    keep_folder_name = f"./record_GA_{model}/ELEMENT/{args.heterogeneity}/-1"
    base_path = f"./record_GA_{model}/ELEMENT/{args.heterogeneity}"
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)  
        if os.path.isdir(item_path) and item != keep_folder_name:
            shutil.rmtree(item_path)  
    
    for args_list in tile_mapping_set:
      if args_list!=[]:
        for comb_set in args_list[0]:
          parallel_list.append((idx_cnt, comb_set))
          idx_cnt+=1
    for comb_set in best_result.keys():
      parallel_list.append((idx_cnt, eval(comb_set)))
      idx_cnt+=1
    print("Start compute PPA last", len(parallel_list))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
      executor.map(internal_parallel, parallel_list)

    if args.search_accuracy== 1:
      for parallel in parallel_list:
        temp_list = []
        for ind,model in enumerate(model_list): 
          arch_configs = parallel[1][ind][0]
          selection_str = parallel[1][ind][1]
          adc_bit = arch_configs[0][-2]
          cell_bit = arch_configs[0][-1]
          layerParams, sizes, onnx_model, quantize, adc_ranges_cadidate, index_row_col, dac_ranges, positiveInputsOnly, xy_pars, weight_for_quantized, config = parameter_crosssim[f"{model},{adc_bit},{cell_bit}"] 
          selection = [int(num) for num in selection_str]
          reconstructed_list = [index_row_col.index([arch_configs[num][0],arch_configs[num][1]]) for num in selection]
          init_idx = []
          init_adc_idx = []
          mvm = 0
          Nlayers = len(layerParams)
          for j in range(Nlayers):
            if layerParams[j]['type'] in ('conv'):
              if layerParams[j]["depthwise"] is True:
                init_idx.append(None)
              else:
                init_idx.append(reconstructed_list[mvm])
                init_adc_idx.append(0)
                mvm += 1
            elif layerParams[j]['type'] in ('dense'):
              init_idx.append(reconstructed_list[mvm])
              init_adc_idx.append(0)
              mvm += 1
          accuracy = (run_inference_for_search.calculate_fitness_with_simulation(init_idx,init_adc_idx,layerParams, sizes, onnx_model, quantize, adc_ranges_cadidate, index_row_col, dac_ranges, positiveInputsOnly, xy_pars, weight_for_quantized,config))
          # print(accuracy)
          temp_list.append(results[str(parallel[1])][ind]+ (accuracy[0]*100,))
        results[str(parallel[1])] = temp_list
    
  print("results", results)
  # print("tile_mapping_set", tile_mapping_set)
  tmp_dic=results

  #here we now write code to merge and TOPSIS tmp_dic to config looking for performance
  topsis_point = []
  for value in tmp_dic.values():
    if args.search_accuracy == 0:
      point = (0,0,0)
    else :
      point = (0,0,0,0)
    area_tmp = 0
    
    if len(value)>1:
      for ind in range(len(model_list)):
        if area_tmp < value[ind][2]:
          area_tmp = value[ind][2]
        point = tuple(a + b for a, b in zip(point, value[ind]))
      
      point = list(point)
      point[2] = area_tmp
      topsis_point.append(point)

  current_gen_len = len(topsis_point)

  if cnt>1 and (cnt < args.generation and nostop):
    for value in best_result.values():
      if args.search_accuracy == 0: 
        point = (0,0,0)
      else:
        point = (0,0,0,0)
      area_tmp = 0
      for ind in range(len(model_list)):
        if area_tmp < value[ind][2]:
          area_tmp = value[ind][2]
        point = tuple(a + b for a, b in zip(point, value[ind]))
      
      point = list(point)
      point[2] = area_tmp
      topsis_point.append(point)
    tmp_dic = {**tmp_dic, **best_result}
  print(topsis_point)
  if args.search_accuracy == 1:
    if cnt < args.generation and nostop:
      w = [args.latency,args.power,args.area,args.accuracy]
      sign = np.array([False,False,False,False])
    else:
      w = [args.latency,args.power,args.area,args.accuracy]
      sign = np.array([False,False,False,True])
  else:
    w = [args.latency,args.power,args.area]
    sign = np.array([False,False,False])
  t = Topsis(topsis_point, w, sign)
  t.calc()

	## compare and crop here
  best_result_tmp = [ [] for i in range(len(all_data[model]))]
  if cnt == 1:
    all_set = list(tmp_dic.keys())
  else:
    all_set = list(tmp_dic.keys()) + list(best_result.keys())
  
  check_cnt = 0


  if cnt < args.generation and nostop:
    cand_all = [ [[] for i in range(len(all_data[model]))] for model in model_list]
    check_perf = [0 for i in range(len(all_data[model]))]
    for beam in range(len(t.rank_to_best_similarity())):
      if beam <= len(t.rank_to_best_similarity())/2+1:
        value = all_set[t.rank_to_best_similarity()[beam]-1]
        value = eval(value)
        if best_result_tmp[config_order[model_list[0]].index(value[0][0])] == []:
          best_result_tmp[config_order[model_list[0]].index(value[0][0])].append(value)
          best_result_tmp[config_order[model_list[0]].index(value[0][0])].append(tmp_dic[str(value)])
        for index, model in enumerate(model_list):
          try:
            cand_all[index][config_order[model].index(value[index][0])].append(value[index][1])
          except Exception as e:
            print(index)
            print(value)
      
        if cnt>1 and t.rank_to_best_similarity()[beam] > current_gen_len and check_perf[config_order[model_list[0]].index(value[0][0])] == 0:
          check_cnt += 1
        
        check_perf[config_order[model_list[0]].index(value[0][0])] += 1

      else:
        value = all_set[t.rank_to_best_similarity()[beam]-1]
        value = eval(value)
        if best_result_tmp[config_order[model_list[0]].index(value[0][0])] == []:
          best_result_tmp[config_order[model_list[0]].index(value[0][0])].append(value)
          best_result_tmp[config_order[model_list[0]].index(value[0][0])].append(tmp_dic[str(value)])
        for index, model in enumerate(model_list):
          if len(cand_all[index][config_order[model].index(value[index][0])]) == 0:
            cand_all[index][config_order[model].index(value[index][0])].append(value[index][1])
        
        if cnt>1 and t.rank_to_best_similarity()[beam] > current_gen_len and check_perf[config_order[model_list[0]].index(value[0][0])] == 0:
          check_cnt += 1
        
        check_perf[config_order[model_list[0]].index(value[0][0])] += 1

    ##  change the check method
    if stop_count > 0 and (check_cnt / len(all_data[model])) <= 0.3:
      stop_count = 0

    if check_cnt / len(all_data[model]) > 0.3:
      stop_count += 1

    if stop_count >= 3:
      nostop = False

  else:
    cand_result=[[] for i in model_list]
    for beam in range(int(len(t.rank_to_best_similarity())/2+1)):
      value = all_set[t.rank_to_best_similarity()[beam]-1]
      value = eval(value)
      for index, model in enumerate(model_list):
        if str(value) in tmp_dic.keys():
          cand_result[index].append([value[index][0],value[index][1],tmp_dic[str(value)][index]])
    break

  print("Cand", cand_all)

  tile_mapping_set = [[] for i in range(len(tmp[0]))]
  results = Manager().dict()
  best_tmp = {}
  for value in best_result_tmp:
    if value != []:
      best_tmp[str(value[0])] = value[1]

  best_result = best_tmp
  print(best_result)
  cnt+=1
  

print("cand", cand_result)

if not os.path.isdir(f"{navcim_dir}/Inference_pytorch/search_result/{args.models}_model_search_result_GA_predict"):
  os.makedirs(f"{navcim_dir}/Inference_pytorch/search_result/{args.models}_model_search_result_GA_predict")

for index, model in enumerate(model_list):
  if args.search_accuracy == 1:
    final_latency_f = open(f"./search_result/{args.models}_model_search_result_GA_predict/final_result_{model}_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area},{args.accuracy}]_{args.heterogeneity}_{args.generation}_{args.population_size}_GA.txt",'w', newline='')
  else:
    final_latency_f = open(f"./search_result/{args.models}_model_search_result_GA_predict/final_result_{model}_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area}]_{args.heterogeneity}_{args.generation}_{args.population_size}_GA.txt",'w', newline='')
  final_latency_wr = csv.writer(final_latency_f)
  for cand in cand_result[index]:
    print(cand)
    final_latency_wr.writerow([cand[0],cand[1],cand[2]])
  final_latency_f.close()
  
  
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")



########

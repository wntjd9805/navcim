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
from multiprocessing import Pool,Manager,Value, Lock
from concurrent.futures import ProcessPoolExecutor
import itertools
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
import torch.nn as nn
import joblib
from sklearn.preprocessing import PolynomialFeatures
import pickle
import pandas as pd

from sys import path
navcim_dir = os.getenv('NAVCIM_DIR')
path.append(f"{navcim_dir}/TOPSIS-Python/")
from topsis import Topsis
# torch.set_num_threads(80)


def time_measure(start_time=None):
    if not start_time:
        return time.time()
    else:
        return time.time() - start_time


parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--model', default='ResNet50', help='VGG16|ResNet50|NasNetA|LFFD')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 64)')
parser.add_argument('--heterogeneity', type=int, default=2, help='heterogeneity')
parser.add_argument('--distribute',type=int ,default=1, help='distribute')
parser.add_argument('--beam_size_m',type=int ,default=700,help='beam_size_m')
parser.add_argument('--beam_size_n',type=int ,default=3,help='beam_size_n')
parser.add_argument('--latency',type=int ,default=1,help='weight_latency_with_pareto')
parser.add_argument('--power',type=int ,default=1,help='weight_power_with_pareto')
parser.add_argument('--area',type=int ,default=1,help='weight_area_with_pareto')
parser.add_argument('--accuracy',type=int ,default=1,help='weight_accuracy_with_pareto')
parser.add_argument('--search_accuracy',type=int ,default=0, help='search_accuracy')
parser.add_argument('--search_accuracy_metric', type=str, default='cka', choices=['mse', 'cosine', 'ssim', 'cka'], help='metric')
parser.add_argument('--constrain_latency',type=int ,default=17506658,help='constrain value')
parser.add_argument('--constrain_power',type=int ,default=122,help='constrain value')
parser.add_argument('--constrain_area',type=int ,default=43306398,help='constrain value')

args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'


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


predict_model_dict={}

for adc in adc_set:
  for cellbit in cellbit_set:
    #path
    loaded_model_filename = f'{navcim_dir}/Inference_pytorch/predict_model/{args.model}/polynomial_regression_{args.model}_latency_{adc}_{cellbit}final.pkl'
    loaded_poly_filename = f'{navcim_dir}/Inference_pytorch/predict_model/{args.model}/polynomial_features_{args.model}_latency_{adc}_{cellbit}final.pkl'
    with open(loaded_model_filename, 'rb') as file:
        linear_model = joblib.load(file)
    with open(loaded_poly_filename, 'rb') as file:
        polynomial_features = joblib.load(file) 

    loaded_model_filename_energy = f'{navcim_dir}/Inference_pytorch/predict_model/{args.model}/mlp_{args.model}_energy_{adc}_{cellbit}_scale_final.pkl'
    loaded_scaler_filename_energy = f'{navcim_dir}/Inference_pytorch/predict_model/{args.model}/mlp_scaler_x_{args.model}_energy_{adc}_{cellbit}final.pkl'
    loaded_scaler_y_filename_energy = f'{navcim_dir}/Inference_pytorch/predict_model/{args.model}/mlp_scaler_y_{args.model}_energy_{adc}_{cellbit}final.pkl'

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
    
    predict_model_dict[f"{adc},{cellbit}"] = [linear_model,polynomial_features ,loaded_scaler_energy,scaler_x_scaler_energy,scaler_y_scaler_energy]
def xy_routing(start_node, end_node):
    """
    Compute the XY routing path in a 2D mesh network.
    
    :param start_node: Tuple (x, y) indicating the starting node position.
    :param end_node: Tuple (x, y) indicating the ending node position.
    :return: List of tuples representing the path from start to end node.
    """
    path = []
    current_node = start_node

    # Move in the X direction until the X coordinate of the end node is reached
    while current_node[0] != end_node[0]:
        if current_node[0] < end_node[0]:
            current_node = (current_node[0] + 1, current_node[1])
        else:
            current_node = (current_node[0] - 1, current_node[1])
        path.append(current_node)

    # Move in the Y direction until the Y coordinate of the end node is reached
    while current_node[1] != end_node[1]:
        if current_node[1] < end_node[1]:
            current_node = (current_node[0], current_node[1] + 1)
        else:
            current_node = (current_node[0], current_node[1] - 1)
        path.append(current_node)

    return path

def generate_snake_mapping(node_count, network_size):
    """
    Generate a list of node positions in a snake-like mapping for a given node count and network size.

    :param node_count: Number of nodes to map.
    :param network_size: Tuple indicating the size of the network (rows, columns).
    :return: List of tuples representing node positions.
    """
    nodes = []
    rows, cols = network_size
    for i in range(node_count):
        row = i // cols
        col = i % cols if row % 2 == 0 else (cols - 1) - (i % cols)
        nodes.append((row, col))
    return nodes

def generate_snake_mapping_for_given_nodes(start_node_count, end_node_count, network_size):
    """
    Generate snake mapping for the given number of start and end nodes in a 2D mesh network.

    :param start_node_count: Number of start nodes.
    :param end_node_count: Number of end nodes.
    :param network_size: Tuple indicating the size of the network (rows, columns).
    :return: Two lists of tuples, representing the positions of start and end nodes.
    """
    total_nodes = start_node_count + end_node_count
    all_nodes = generate_snake_mapping(total_nodes, network_size)

    # Split the nodes into start and end nodes
    start_nodes = all_nodes[:start_node_count]
    end_nodes = all_nodes[start_node_count:start_node_count + end_node_count]

    return start_nodes, end_nodes

def calculate_average_hops(start_node_count, end_node_count, network_size):
    """
    Calculate the average number of hops for snake-mapped start and end nodes in a 2D mesh network.

    :param start_node_count: Number of start nodes.
    :param end_node_count: Number of end nodes.
    :param network_size: Tuple indicating the size of the network (rows, columns).
    :return: Average number of hops for all the paths.
    """
    # Generate snake mapping for start and end nodes
    start_nodes, end_nodes = generate_snake_mapping_for_given_nodes(start_node_count, end_node_count, network_size)

    # Calculate the total hops and the total number of paths
    total_hops = 0
    total_paths = len(start_nodes) * len(end_nodes)

    for start_node in start_nodes:
        for end_node in end_nodes:
            path = xy_routing(start_node, end_node)
            total_hops += len(path)

    # Calculate the average number of hops
    return total_hops / total_paths if total_paths > 0 else 0

def inverse_minmax_latency(value, max, min):
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
        self.fc7 = nn.Linear(32, 16)  
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
        self.fc5 = nn.Linear(32, 16)
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

def injection_rate(activation_size, Nbit, FPS, bus_width, freq):
  rate = (activation_size * Nbit * FPS) / (bus_width * freq)
  return rate


def comb2(arr):
    result = []
    for i in range(len(arr)):
        for j in arr[i + 1:]:
            result.append((arr[i], j))
    return result

def lcm(a,b):
  return (a * b) // math.gcd(a,b)

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


def distribute(row,col,data_in,data_out, CLUSTER):
  data_intra_spread = data_in/row
  data_intra_gather = data_out/col
  
  minimum_split_row=math.ceil(row/CLUSTER)
  minimum_split_col=math.ceil(col/CLUSTER)
  candidate_row= get_divisor(row, minimum_split_row)
  candidate_col= get_divisor(col, minimum_split_col)
  result= None
  final_cost=None
  for split_row in candidate_row:
    for split_col in candidate_col:
      num_of_tile_in_cluster = (row/split_row) * (col/split_col)
      cost = data_in*split_col+(data_intra_gather+data_intra_spread)*num_of_tile_in_cluster+data_out*split_row
      if final_cost is None or cost < final_cost:
         final_cost = cost
         result = [split_row,split_col]
  
  return result
         
   
def make_file_name(config):
  file_name = ""
  for i in config:
    file_name += f"ADC{i[4]}_Cellbit{i[5]}_SArow{i[0]}_SAcol{i[1]}_PE{i[2]}_TL{i[3]}_"
  return file_name
##place and rounting
def profile_for_select(i , tmp_dfg, Nbit ,Bus_width, chip_clk_freq, cluster_clk_freq, FPS ,CLUSTER, iter, cluster_wr, element_wr, shape, level, count,mapping_num_list , chip_width, chip_number,tile_grid, mapping_info):
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
    if args.model == "VGG8" or args.model == "DenseNet40":
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
  return iter, count,mapping_num_list , level, tile_grid, num_of_col ,addition_cluster
  #----

def profile_for_select_distribute(i , tmp_dfg, Nbit ,Bus_width, chip_clk_freq, cluster_clk_freq, FPS ,CLUSTER, iter, cluster_wr,element_wr, shape, level, count,mapping_num_list,chip_width,chip_number,tile_grid, mapping_info):
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
    if args.model == "VGG8" or args.model == "DenseNet40":
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
  
  split = distribute(row,col,numBitToLoadIn,numBitToLoadOut,CLUSTER)

  num_of_row = split[0]
  num_of_col = split[1]
  number_of_cluster = num_of_row * num_of_col
  num_tile_per_cluster = (row/num_of_row) *(col/num_of_col)
  
  addition_cluster = 0
  injection = injection_rate(numBitToLoadIn/num_of_row, Nbit, FPS, Bus_width, cluster_clk_freq)
  mapping_num_list.append(int(number_of_cluster))
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
    
    if int(i) < 2:
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
    if int(i) < 2:
      element_wr.writerow([name_of_element_tile,tmp_str1,(numBitToLoadIn/row),injection_element_send,(numBitToLoadOut/col),injection_element_receive,row,col])
  
    if exist == 1:
      mapping_info[f"{target[0]}-{target[1]}"][2] = node_in_tile
      mapping_info[f"{target[0]}-{target[1]}"][3] = mapping_info[f"{target[0]}-{target[1]}"][3] - int(num_tile_per_cluster)
    else:
      mapping_info[f"{target[0]}-{target[1]}"]=[[i],chip_number,node_in_tile,(CLUSTER**2)-int(num_tile_per_cluster)]
  

  st1 = node[i][1]
  st2 = node[i][2]

  key_position1 = node_keys_list.index(st1) if st1 in node_keys_list else None
  key_position2 = node_keys_list.index(st2) if st2 in node_keys_list else None

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

  return iter, count,mapping_num_list , level, tile_grid, num_of_col ,addition_cluster ,list_for_predict
    

  
def number_of_tile_sofar(mapping_info):
  num_chip1=0
  num_chip2=0
  for t in mapping_info.values():
    if t[1] == 0:
      num_chip1+=1
    elif t[1] == 1:
      num_chip2 +=1
  return num_chip1,num_chip2
  
def PPA_function(latency, energy, area):
  return latency+energy/latency+area

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

def execute_booksim(node, cluster_width, chip1_width, cluster_flit_cycle, chip1_flit_cycle,model,select, cluster_meter, chip1_meter, chip_period,cluster_period, cluster_buswidth ,chip1_buswidth, file_name, NUM_ROW_cluster, NUM_COL_cluster):
  cmd1 = f'{navcim_dir}/booksim2/src/booksim {navcim_dir}/booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {cluster_flit_cycle} {chip1_flit_cycle} {navcim_dir}/Inference_pytorch/record_predict_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{model}_{file_name}{select}.txt {navcim_dir}/Inference_pytorch/record_predict_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{model}_{file_name}{select}.txt {cluster_meter} {chip1_meter} {cluster_buswidth} {chip1_buswidth} 0 na {node} 1 | egrep "taken|Total Power|Total Area|Total leak Power" > ./record_predict_{args.model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/BOOKSIM_{model}_{file_name}{select}.txt'
  cmd2 = f'{navcim_dir}/booksim2/src/booksim {navcim_dir}/booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {cluster_flit_cycle} {chip1_flit_cycle} {navcim_dir}/Inference_pytorch/record_predict_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{model}_{file_name}{select}.txt {navcim_dir}/Inference_pytorch/record_predict_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{model}_{file_name}{select}.txt {cluster_meter} {chip1_meter} {cluster_buswidth} {chip1_buswidth} 0 na {node} 2 | egrep "taken|Total Power|Total Area|Total leak Power" >> ./record_predict_{args.model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/BOOKSIM_{model}_{file_name}{select}.txt'
  cmd3 = f'{navcim_dir}/booksim2/src/booksim {navcim_dir}/booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {cluster_flit_cycle} {chip1_flit_cycle} {navcim_dir}/Inference_pytorch/record_predict_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{model}_{file_name}{select}.txt {navcim_dir}/Inference_pytorch/record_predict_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{model}_{file_name}{select}.txt {cluster_meter} {chip1_meter} {cluster_buswidth} {chip1_buswidth} 0 na {node} 3 | egrep "taken|Total Power|Total Area|Total leak Power" >> ./record_predict_{args.model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/BOOKSIM_{model}_{file_name}{select}.txt'
  try:
    output = subprocess.check_output(
        cmd1, stderr=subprocess.STDOUT, shell=True)
    
  except subprocess.CalledProcessError as exc:
      print("Error!!!!!", exc.returncode, exc.output, cmd1)
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
  
  
  fname = f"./record_predict_{args.model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/BOOKSIM_{model}_{file_name}{select}.txt"
  latency_result = 0
  energy_result=0
  area_cluster=0
  area_chip = 0 
  cluster_leak_power = 0
  chip_leak_power = 0
  with open(fname) as f:
    lines = f.readlines()
    latency_result = int(lines[0].split("\n")[0].split(" ")[3])* cluster_period + int(lines[4].split("\n")[0].split(" ")[3])*chip_period + int(lines[8].split("\n")[0].split(" ")[3])*chip_period
    energy_result = float(lines[1].split("\n")[0].split(" ")[3])* int(lines[0].split("\n")[0].split(" ")[3])* cluster_period+ float(lines[5].split("\n")[0].split(" ")[3])*int(lines[4].split("\n")[0].split(" ")[3])*chip_period  + float(lines[9].split("\n")[0].split(" ")[3])*int(lines[8].split("\n")[0].split(" ")[3])*chip_period
    area_cluster = float(lines[2].split("\n")[0].split(" ")[3])*1000000
    area_chip = float(lines[6].split("\n")[0].split(" ")[3])*1000000
    cluster_leak_power = float(lines[3].split("\n")[0].split(" ")[4])
    chip_leak_power = float(lines[7].split("\n")[0].split(" ")[4])

  return latency_result,energy_result*1000,area_cluster,area_chip,cluster_leak_power,chip_leak_power


def simple_cull(inputPoints, dominates):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints

def dominates(row, candidateRow):
  for i in range(len(row)):
    if args.search_accuracy ==1 and i == len(row)-1:
      if row[i] > candidateRow[i]:
        return False
    else:
      if row[i] >= candidateRow[i]*0.95:
        return False
  return True   
def keep_efficient(input_set):
  # sort points by decreasing sum of coordinates
  start = time.time()
  pts = np.array(list(input_set))
  pts = pts[pts.sum(1).argsort()]
  # initialize a boolean mask for undominated points
  # to avoid creating copies each iteration
  undominated = np.ones(pts.shape[0], dtype=bool)
  for i in range(pts.shape[0]):
      # process each point in turn
      n = pts.shape[0]
      if i >= n:
          break
      # find all points not dominated by i
      # since points are sorted by coordinate sum
      # i cannot dominate any points in 1,...,i-1
      undominated[i+1:n] = (pts[i+1:]*0.95 <= pts[i]).any(1) 
      # keep points undominated so far
      pts = pts[undominated[:n]]
  print("pareto time",time.time()-start)
  return set(tuple(point) for point in pts)

def initialization(config):
  num_tile = {}
  FPS_tmp_dict ={}
  shape_tmp_dict  =[]
  PPA_tmp_dict  =[]
  
  file_name = make_file_name(config)
  for cf in range(len(config)):
    NUM1_ROW = config[cf][0]
    NUM1_COL = config[cf][1]
    PE1 = config[cf][2]
    Tile1 = config[cf][3]
    ADC = config[cf][4]
    Cellbit=config[cf][5]

    shape_tmp={}
    fname1 = f"./shape/shape_{args.model}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"
    with open(fname1) as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            l=l.replace("\n", "")
            shape_tmp[i]=l.split(',')


    CLUSTER1 = int(math.ceil((NUM1_ROW*Tile1)/(config[0][0]*config[0][3])))
    compute_PPA_tmp={}
    FPS_latency = 0
    for layer_num, layer in enumerate(All_layer):
      latency1 = LATENCY.get(f"{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")-inter_connect_LATENCY.get(f"{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
      energy1 = POWER.get(f"{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")-inter_connect_POWER.get(f"{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
      area1 = AREA.get(f"{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")-inter_connect_AREA.get(f"{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
      compute_PPA_tmp[layer_num] = [latency1,energy1,area1] 
      FPS_latency = FPS_latency + latency1*1e-9
      if layer_num in num_tile:
        num_tile[layer_num] = max( math.ceil(int(shape_tmp[layer_num][4])/(CLUSTER1**2)), num_tile[layer_num])
      else :
        num_tile[layer_num] = math.ceil(int(shape_tmp[layer_num][4]) / (CLUSTER1**2))
    shape_tmp_dict.append(shape_tmp)
    PPA_tmp_dict.append(compute_PPA_tmp)
    FPS_tmp_dict[cf] = 1/FPS_latency
    
    tail_distribute = ""

  cluster_f = open(f"./record_predict_{args.model}/CLUSTER/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}/CLUSTER_{args.model}_{file_name}{tail_distribute}.txt",'w', newline='')
  cluster_wr = csv.writer(cluster_f)
  cluster_wr.writerow(["node","destination1","destination2","op","location","type","activation_size","injection_rate"])
  cluster_f.close()

  element_f = open(f"./record_predict_{args.model}/ELEMENT/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}/ELEMENT_{args.model}_{file_name}{tail_distribute}.txt",'w', newline='')
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
  mapping_num_info = []
  tmp_dfg = dfg
  mapping_info = {}
  conv_dense=0
  total_latency=0
  total_energy=0
  total_area=0
  total_leakage=0
  selected_list=[]
  area_leak_dic = {}
  
  del num_tile
  del FPS_tmp_dict

  return shape_tmp_dict ,PPA_tmp_dict ,chip_width,FPS,iter,level,count,mapping_num_info,tile_grid,mapping_info,conv_dense,total_latency,total_energy,total_area,total_leakage,selected_list,area_leak_dic
  


def make_args(ht,config,config_cluster,tmp_dfg,shape1,compute_PPA1,chip_width,FPS,iter,level,count,mapping_num_list,tile_grid,mapping_info,conv_dense,total_latency,total_energy,total_area,total_leakage,total_similarity,dfg_key,selected_list,file_name,Booksim_Leakage_Area):
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
  

  linear_model = predict_model_dict[f"{ADC},{Cellbit}"][0]
  polynomial_features = predict_model_dict[f"{ADC},{Cellbit}"][1]
  loaded_scaler_energy = predict_model_dict[f"{ADC},{Cellbit}"][2]
  scaler_x_scaler_energy = predict_model_dict[f"{ADC},{Cellbit}"][3]
  scaler_y_scaler_energy = predict_model_dict[f"{ADC},{Cellbit}"][4]
  
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
    tail_select = tail_select + str(selected)+"_"
  
  tail_distribute = tail_select
  
  Nbit=8
  node_col={}
  inverse = {}
  tail1=f"{tail_select}{ht}_"
  
  addition_cluster = 0
  for i in dfg_key:
    # print(f"---------------------{i}---------------------")
    # print(tmp_dfg[str(i)][0])  
    if (tmp_dfg[str(i)][0]=='nn.conv2d' or tmp_dfg[str(i)][0]=='nn.dense'):
      os.makedirs(f"./record_predict_{args.model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}", exist_ok=True)
      os.makedirs(f"./record_predict_{args.model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}", exist_ok=True)
      os.makedirs(f"./record_predict_{args.model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}", exist_ok=True)
      shutil.copy(f"./record_predict_{args.model}/CLUSTER/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail_distribute}.txt" ,f"./record_predict_{args.model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}.txt")
      shutil.copy(f"./record_predict_{args.model}/ELEMENT/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{args.model}_{file_name}{tail_distribute}.txt" ,f"./record_predict_{args.model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{args.model}_{file_name}{tail1}.txt")
      iter1 = copy.deepcopy(iter)
      level1 = copy.deepcopy(level)
      count1 = copy.deepcopy(count)
      mapping_num_list1 = copy.deepcopy(mapping_num_list)
      tile_grid1 = copy.deepcopy(tile_grid)
      mapping_info1 = copy.deepcopy(mapping_info)
      selected_list1= copy.deepcopy(selected_list)
      copy_lock = 1

      cluster_f_select1 = open(f"./record_predict_{args.model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}.txt",'a')
      cluster_wr_select1 = csv.writer(cluster_f_select1)
      element_f_select1 = open(f"./record_predict_{args.model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{args.model}_{file_name}{tail1}.txt",'a')
      element_wr_select1 = csv.writer(element_f_select1)
      num_of_col1 = None
      booksim_energy_cluster = 0
      booksim_latency_cluster = 0
      if args.distribute == 1:
        iter1, count1,mapping_num_list1, level1, tile_grid1,num_of_col1,addition_cluster, list_for_predict= profile_for_select_distribute(i, tmp_dfg , Nbit ,chip1_buswidth, chip1_clk_freq, cluster_clk_freq, FPS ,CLUSTER1, iter1, cluster_wr_select1 ,element_wr_select1, shape1.get(conv_dense), level1, count1, mapping_num_list1,chip_width, ht, tile_grid1,mapping_info1)
      else:
        iter1, count1, mapping_num_list1 ,level1, tile_grid1,num_of_col1,addition_cluster= profile_for_select(i, tmp_dfg , Nbit ,chip1_buswidth, chip1_clk_freq, cluster_clk_freq, FPS ,CLUSTER1, iter1, cluster_wr_select1 ,element_wr_select1, shape1.get(conv_dense), level1, count1, mapping_num_list1,chip_width, ht, tile_grid1,mapping_info1)
 
      cluster_f_select1.close()
      element_f_select1.close()
      # print(f'--------------layer{conv_dense}-----------------')
      # print(f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}')
      # print("compute latency", compute_PPA1[conv_dense])
      if depth==0: #or depth ==1:
        os.makedirs(f"./record_predict_{args.model}/Prepared_data/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}", exist_ok=True)
        os.makedirs(f"./record_predict_{args.model}/Prepared_data/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}", exist_ok=True)
        booksim_latency1,booksim_energy1,booksim_latency_cluster,booksim_energy_cluster ,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage = predict_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,args.model,tail1,cluster_meter,chip1_meter,list_for_predict,1)
        Booksim_Leakage_Area[ht] = [cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage]
        
      else:
        booksim_latency1,booksim_energy1,booksim_latency_cluster,booksim_energy_cluster = predict_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,args.model,tail1,cluster_meter,chip1_meter,list_for_predict,0)
        cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage =  Booksim_Leakage_Area[ht]

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
        total_energy1 = total_energy + scale_energy[0][0] #+ compute_PPA1[conv_dense][1]
        if depth == 0 :
          total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2))*chip_width + (cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
          total_area1 = total_area + cluster_booksim_area + (chip1_booksim_area+cluster_area)*addition_cluster

        else :
          total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
          total_area1 = total_area + (chip1_booksim_area+cluster_area)*addition_cluster
        selected_list1.append(ht)
        
     
        conv_dense=conv_dense+1

        if args.search_accuracy == 1:
          total_similarity1 = total_similarity + (Similarity_for_accuracy[f"{ADC},{Cellbit}"][f"{NUM1_ROW},{NUM1_COL}"][conv_dense-1]* math.pow(hessian_list[conv_dense-1],2))
          return iter1,level1,count1,mapping_num_list1,tile_grid1,mapping_info1,conv_dense,float(total_latency1),float(total_energy1),total_area1,total_leakage1,selected_list1 ,Booksim_Leakage_Area,total_similarity1
        
        else:
          return iter1,level1,count1,mapping_num_list1,tile_grid1,mapping_info1,conv_dense,float(total_latency1),float(total_energy1),total_area1,total_leakage1,selected_list1,Booksim_Leakage_Area,0

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
      cluster_f_select1 = open(f"./record_predict_{args.model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}.txt",'a')
      cluster_wr_select1 = csv.writer(cluster_f_select1)
      cluster_wr_select1.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{iter1[0]}-0","non_MAC",key_loc_before_size/num_of_col1,injection1,1,1])
      cluster_f_select1.close()
      mapping_num_list1.append(1)
      booksim_latency_non_mac = 0
      booksim_energy_non_mac = 0
      
      st1 = node[i][1]
      st2 = node[i][2]

      key_position1 = node_keys_list.index(st1) if st1 in node_keys_list else None
      key_position2 = node_keys_list.index(st2) if st2 in node_keys_list else None

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

      booksim_latency_non_mac,booksim_power_non_mac, booksim_latency_non_mac_cluster,booksim_energy_non_mac_cluster = predict_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,args.model,tail1,cluster_meter,chip1_meter, list_for_predict,0)
    
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
          total_energy1 = total_energy + scale_energy[0][0] #+ before_conv_result1[1][1]
       
      
      if depth == 0 :
        total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2))*chip_width + (cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
        total_area1 = total_area + cluster_booksim_area + (chip1_booksim_area+cluster_area)*addition_cluster
      else :
        total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
        total_area1 = total_area + (chip1_booksim_area+cluster_area)*addition_cluster
      
      selected_list1.append(ht)
     
      if args.search_accuracy == 1:
        total_similarity1 = total_similarity + (Similarity_for_accuracy[f"{ADC},{Cellbit}"][f"{NUM1_ROW},{NUM1_COL}"][conv_dense-1] * math.pow(hessian_list[conv_dense-1],2))
        return iter1,level1,count1,mapping_num_list1,tile_grid1,mapping_info1,conv_dense,float(total_latency1),float(total_energy1),total_area1,total_leakage1,selected_list1,Booksim_Leakage_Area,total_similarity1

      else:
        return iter1,level1,count1,mapping_num_list1,tile_grid1,mapping_info1,conv_dense,float(total_latency1),float(total_energy1),total_area1,total_leakage1,selected_list1,Booksim_Leakage_Area,0
def my_func(input):
    config = input[0]
    exec_set =input[1]
    config_origin = find_origin_config[f'{config}']
    file_name = make_file_name(config_origin)
    for ht in range(args.heterogeneity):
      iter1_tmp,level1_tmp,count1_tmp,mapping_num_list_tmp,tile_grid1_tmp,mapping_info1_tmp,conv_dense_tmp,total_latency1_tmp,total_energy1_tmp,total_area1_tmp,leakage1_tmp,selected_list1_tmp ,area_leak_dic_list[f'{config_origin}'], total_similarity_tmp = make_args(ht,config_origin[ht],config_origin[0], dfg, globals()['shape{}'.format(ht)][f'{config_origin}'] , globals()['compute_PPA{}_list'.format(ht)][f'{config_origin}'], chip_width_list[f'{config}'], FPS_list[f'{config}'], iter_list[f'{config}'], level_list[f'{config}'], count_list[f'{config}'], mapping_num[f'{config}'],tile_grid_list[f'{config}'], mapping_info_list[f'{config}'], conv_dense_list[f'{config}'], total_latency_list[f'{config}'], total_energy_list[f'{config}'],total_area_list[f'{config}'],total_leakage_list[f'{config}'],total_similarity_list[f'{config}'],exec_set, all_selected_list[f'{config}'], file_name, area_leak_dic_list[f'{config_origin}'])

      with lock:
        selecte1 = f'_{ht}'
        chip_width_list[f'{config}{selecte1}'] = chip_width_list[f'{config}']
        FPS_list[f'{config}{selecte1}'] = FPS_list[f'{config}']
        iter_list[f'{config}{selecte1}']=iter1_tmp
        level_list[f'{config}{selecte1}']=level1_tmp
        count_list[f'{config}{selecte1}']=count1_tmp
        mapping_num[f'{config}{selecte1}']=mapping_num_list_tmp
        tile_grid_list[f'{config}{selecte1}']=tile_grid1_tmp
        mapping_info_list[f'{config}{selecte1}']=mapping_info1_tmp
        conv_dense_list[f'{config}{selecte1}']=conv_dense_tmp
        total_latency_list[f'{config}{selecte1}']=total_latency1_tmp
        total_energy_list[f'{config}{selecte1}']=total_energy1_tmp
        total_leakage_list[f'{config}{selecte1}']=leakage1_tmp
        total_area_list[f'{config}{selecte1}']=total_area1_tmp
        total_similarity_list[f'{config}{selecte1}']=total_similarity_tmp
        all_selected_list[f'{config}{selecte1}']=selected_list1_tmp
        find_origin_config[f'{config}{selecte1}']=find_origin_config[f'{config}']
        run_list.append(f'{config}{selecte1}')

    
    with lock:
      run_list.remove(f'{config}')
      del chip_width_list[f'{config}']
      del FPS_list[f'{config}']
      del iter_list[f'{config}']
      del level_list[f'{config}']
      del count_list[f'{config}']
      del mapping_num[f'{config}']
      del tile_grid_list[f'{config}']
      del mapping_info_list[f'{config}']
      del conv_dense_list[f'{config}']
      del total_latency_list[f'{config}']
      del total_energy_list[f'{config}']
      del total_area_list[f'{config}']
      del total_leakage_list[f'{config}']
      del total_similarity_list[f'{config}']
      del all_selected_list[f'{config}']
      del find_origin_config[f'{config}']

LATENCY={}
POWER={}
AREA={}
leakage_POWER={}
inter_connect_LATENCY={}
inter_connect_POWER={}
inter_connect_AREA={}
tile_width_meter={}
clk_frequency={}
clk_period={}
unitLatencyRep={}
unitLatencyWire={}
MAX_period = {}
minDist={}
busWidth={}
ChipArea={}
Booksim_Leakage_Area={}
All_layer = []
Similarity_for_accuracy = {}

CONFIG = []
for sa_row in sa_set:
  for sa_col in sa_set:
    for pe in range(2,pe_set+1):
      for tile in range(4,tile_set+1):
        if tile > pe and tile%pe==0:
          for adc in adc_set:
            for cellbit in cellbit_set:
              if sa_row * pe >= sa_col and sa_row *tile >= sa_col and sa_col * pe >= sa_row and sa_col * tile >= sa_row:
                  CONFIG.append([sa_row,sa_col, pe, tile,adc, cellbit])

if args.search_accuracy == 1:
  #raise error
  name_model = args.model
  for adc in adc_set:
    for cellbit in cellbit_set:
      df = pd.read_csv(f'{navcim_dir}/cross-sim/applications/dnn/inference/{name_model}_{args.search_accuracy_metric}_ADC:{adc}_CellBit{cellbit}_list.csv')
      Similarity_for_accuracy_tmp = {col: df[col].tolist() for col in df.columns}
      Similarity_for_accuracy[f"{adc},{cellbit}"] = Similarity_for_accuracy_tmp
  with open(f'{navcim_dir}/cross-sim/applications/dnn/inference/{args.model}_hessian_list.txt', 'r') as file:
    content = file.read()  
  hessian_list = [float(number) for number in content.split(', ')]

for config in CONFIG:
  NUM_ROW = config[0]
  NUM_COL = config[1]
  PE = config[2]
  Tile = config[3]
  ADC = config[4]
  Cellbit = config[5]
  fname = f"./summary/summary_{args.model}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}.txt"
  with open(fname) as f:
    lines = f.readlines()
    for i, l in enumerate(lines):
      if l.find("readLatency is:")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        latency = l.split(":")[1]
        latency = float(latency[:-3])
        LATENCY[f'{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=latency
        if layername not in All_layer:
          All_layer.append(layername)
      elif l.find("s readDynamicEnergy is:")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        readDynamicEnergy = l.split(":")[1]
        readDynamicEnergy = float(readDynamicEnergy[:-3])
        POWER[f'{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=readDynamicEnergy
      elif l.find("leakagePower is:")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        leakagePower = l.split(":")[1]
        leakagePower = float(leakagePower[:-3])
        leakage_POWER[f'{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=leakagePower
      elif l.find("Tile Area is:")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        cimarea = l.split(":")[1]
        cimarea = float(cimarea[:-5])
        AREA[f'{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=cimarea
      elif l.find("H-tree Area is")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        inter_area = l.split(":")[1]
        inter_area = float(inter_area[:-5])
        inter_connect_AREA[f'{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=inter_area
      elif l.find("H-tree Latency is")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        inter_latency = l.split(":")[1]
        inter_latency = float(inter_latency[:-3])
        inter_connect_LATENCY[f'{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=inter_latency
      elif l.find("H-tree Energy is")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        inter_energy = l.split(":")[1]
        inter_energy = float(inter_energy[:-3])
        inter_connect_POWER[f'{layername}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=inter_energy
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
assert args.model in ['ResNet50','EfficientNetB0','MobileNetV2','SqueezeNet'], args.model
if args.model == 'ResNet50':
    modelCF = onnx.load("./network/resnet50.onnx")
elif args.model == 'EfficientNetB0':
    modelCF = onnx.load("./network/efficientnet_b0.onnx")
elif args.model == 'MobileNetV2':
    modelCF = onnx.load("./network/mobilenet_v2.onnx")
elif args.model == 'SqueezeNet':
    modelCF = onnx.load("./network/squeezenet.onnx")
else:
    raise ValueError("Unknown model type")




graph, params = tvm.relay.frontend.from_onnx(modelCF, shape=None, dtype='float32', opset=None, freeze_params=True, convert_config=None)
graph_ir = str(graph)
parse_ir = graph_ir.split('\n')
node={}
dfg={}


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
        node[name.strip()]=["depthwise",input.strip(),input2.strip(),activation,kernel_size,stride]
      else:
        node[name.strip()]=[type.strip(),input.strip(),input2.strip(),activation,kernel_size,stride]
last_node=None

for i in node.keys():
  if str(i) != '0':
    if node[str(i)][1] != '':
      if '.' in node[str(i)][1]:
        node[str(i)][1] =  str(int(float(node[str(i)][1])))
    if node[str(i)][2] != '':
      if '.' in node[str(i)][2]:
        node[str(i)][2] =  str(int(float(node[str(i)][2])))

for i in node.keys():
  if node[str(i)][0]=='': 
    for j in node.keys():
      if node[str(j)][1]==str(i):
        node[str(j)][1] =  node[str(i)][1]
        node[str(j)][2] =  node[str(i)][2]

for i in node.keys():
  last_node = i
  if node[str(i)][0]!='nn.dense' and node[str(i)][0]!='nn.conv2d' and node[str(i)][0]!='non-MAC':
    for j in node.keys():
      if node[str(j)][1]==str(i):
        if node[str(j)][0]=="nn.conv2d" or node[str(j)][0]=="nn.dense" :
          node[str(i)][0]="non-MAC"
          break
        else:
          has_two=0
          if node[str(i)][1] != '':
            node[str(j)][1]=str(int(float(node[str(i)][1])))
            has_two=1
          if node[str(i)][2] != '':
            if has_two == 1 and (args.model != "EfficientNetB0"):
              node[str(j)][2]=str(int(float(node[str(i)][2])))
            else:
              node[str(j)][1]=str(int(float(node[str(i)][2])))
      
      if node[str(j)][2]==str(i):
        if node[str(j)][0]=="nn.conv2d" or node[str(j)][0]=="nn.dense" :
          node[str(i)][0]="non-MAC"
          break
        else:
          has_two=0
          if node[str(i)][1] != '':
            node[str(j)][2]=str(int(float(node[str(i)][1])))
            has_two=1
          if node[str(i)][2] != '':
            if has_two == 1 and (args.model != "EfficientNetB0"):
              node[str(j)][1]=str(int(float(node[str(i)][2])))
            else:
              node[str(j)][2]=str(int(float(node[str(i)][2])))

pop_list=[]
for i in node.keys():
    if node[str(i)][0]!='nn.dense' and node[str(i)][0]!='nn.conv2d' and node[str(i)][0]!='non-MAC':
      pop_list.append(i)

for i in pop_list:
    node.pop(str(i),None)


non_MAC_list={}
for i in node.keys():
  if node[str(i)][0]=='non-MAC':
    non_MAC_list[i]=node[str(i)][1:3]

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
    node[str(i)][1] = non_MAC_list[str(i)][0]
    node[str(i)][2] = non_MAC_list[str(i)][1]
  else:
    node[str(i)][1] = non_MAC_list[str(i)][1]
    node[str(i)][2] = non_MAC_list[str(i)][0]


node_keys_list = list(node.keys())
for node_key in node.keys():
    key = node_key
    op= node[node_key][0]
    act= node[node_key][3]
    activation_size = None
    for act_ in act:
        if activation_size == None:
            activation_size = int(act_)
        else:
            activation_size = activation_size * int(act_)
    output1=''
    output2=''
    
    for tmp_key in node.keys():
        if node[tmp_key][1]==node_key:
          
          if output1=='':
              output1=tmp_key
          else:
              output2=tmp_key
        if node[tmp_key][2]==node_key:
            if output1=='':
                output1=tmp_key
            else:
                output2=tmp_key
 
    dfg[key]=[op,output1,output2,act,activation_size,1,node[node_key][4],node[node_key][5]]




path = f"{navcim_dir}/Inference_pytorch/record_predict_{args.model}"
 
print(dfg)

CONFIG_pareto=[]
for adc in adc_set:
  for cellbit in cellbit_set:
    for layer_num in All_layer: 
      x = np.array([])
      y = np.array([])
      z = np.array([])
      labels = np.array([])
      for key in LATENCY:
        if key.find(f"{layer_num}_")!=-1 and f"ADC:{adc}" in key and f"Cellbit:{cellbit}" in key:
          x=np.append(x,LATENCY.get(key,0)-inter_connect_LATENCY.get(key,0))
          y=np.append(y,(POWER.get(key,0)+leakage_POWER.get(key,0)-inter_connect_POWER.get(key,0))/(LATENCY.get(key,0)-inter_connect_LATENCY.get(key,0)))
          z=np.append(z,(AREA.get(key,0)-inter_connect_AREA.get(key,0)))
          labels=np.append(labels,key) 


      point = [list(element) for element in zip(x,y,z)]
      pareto_label={}
      for i,e in enumerate(point):
        if str(e[0])+str(e[1])+str(e[2]) in pareto_label.keys():
              pareto_label[str(e[0])+str(e[1])+str(e[2])].append(labels[i])
        else:
          pareto_label[str(e[0])+str(e[1])+str(e[2])]= [labels[i]]


      paretoPoints = keep_efficient(point)
      for pareto in paretoPoints:
        pareto_set = pareto_label[str(pareto[0])+str(pareto[1])+str(pareto[2])]
        for pareto_i in pareto_set:
          ADC_pareto = int(pareto_i.split(':')[1].split('_')[0])
          Cellbit_preto = int(pareto_i.split(':')[2].split('_')[0])
          SA_ROW_pareto = int(pareto_i.split(':')[3].split('_')[0])
          SA_COL_pareto = int(pareto_i.split(':')[4].split('_')[0])
          PE_pareto = int(pareto_i.split(':')[5].split('_')[0])
          Tile_pareto = int(pareto_i.split(':')[6])

        if [SA_ROW_pareto, SA_COL_pareto, PE_pareto, Tile_pareto,ADC_pareto,Cellbit_preto] not in CONFIG_pareto:
          CONFIG_pareto.append([SA_ROW_pareto, SA_COL_pareto, PE_pareto, Tile_pareto,ADC_pareto,Cellbit_preto])

grouped_configs={}
for config in CONFIG_pareto:
    key = (config[4], config[5])
    if key not in grouped_configs:
        grouped_configs[key] = []
    grouped_configs[key].append(config)

combine_CONFIG = []
for group in grouped_configs.values():
    if len(group) >= args.heterogeneity:
        combine_CONFIG.extend(list(combinations(group, args.heterogeneity)))

random.shuffle(combine_CONFIG)

for j, config in enumerate(combine_CONFIG):
  combine_CONFIG[j] = sorted(config, key=lambda x: -x[0]*x[3])

execute_set=[]
tmp_set=[]
print(dfg.keys())
for dfg_node in dfg.keys():
  if dfg[dfg_node][0] == 'nn.conv2d' or dfg[dfg_node][0] == 'nn.dense':
    if len(tmp_set) > 0:
      execute_set.append(tmp_set)
      tmp_set=[dfg_node]
    else:
      tmp_set=[dfg_node]
  else:
    tmp_set.append(dfg_node)
    execute_set.append(tmp_set)
    tmp_set=[]
execute_set.append(tmp_set)
  
print(execute_set)
print(len(execute_set))

chip_width_list=Manager().dict()
FPS_list=Manager().dict()
# change value dependent choosing tile
iter_list=Manager().dict()
level_list=Manager().dict()
count_list=Manager().dict()
mapping_num=Manager().dict()
tile_grid_list=Manager().dict()
mapping_info_list=Manager().dict()
conv_dense_list=Manager().dict()
total_latency_list=Manager().dict()
total_energy_list=Manager().dict()
total_area_list=Manager().dict()
all_selected_list=Manager().dict()
area_leak_dic_list=Manager().dict()
total_leakage_list=Manager().dict()
run_list = Manager().list()
find_origin_config = Manager().dict()
total_similarity_list=Manager().dict()
for i in range(args.heterogeneity):
  globals()['shape{}'.format(i)] = Manager().dict()
  globals()['compute_PPA{}_list'.format(i)] = Manager().dict()

depth = -1

completed_count = Value('i', 0)
lock_w = Lock()
total_tasks = 0
def update_progress(result):
    with lock_w:
        completed_count.value += 1
        if completed_count.value % 100 == 0:  # 100개 작업마다 진행 상황을 출력
            print(f"{completed_count.value}/{total_tasks} tasks completed.")


def init(config):
  os.makedirs(f"{path}/CLUSTER/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}", exist_ok=True)
  os.makedirs(f"{path}/ELEMENT/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}", exist_ok=True)
  shape_list, PPA_list ,chip_width_list[f'{config}'],FPS_list[f'{config}'],iter_list[f'{config}'],level_list[f'{config}'],count_list[f'{config}'],mapping_num[f'{config}'],tile_grid_list[f'{config}'],mapping_info_list[f'{config}'],conv_dense_list[f'{config}'],total_latency_list[f'{config}'],total_energy_list[f'{config}'],total_area_list[f'{config}'],total_leakage_list[f'{config}'],all_selected_list[f'{config}'],area_leak_dic_list[f'{config}'] = initialization(config)
  total_similarity_list[f'{config}'] = 0
  for i in range(args.heterogeneity):
    globals()['shape{}'.format(i)][f'{config}'] = shape_list[i].copy()
    globals()['compute_PPA{}_list'.format(i)][f'{config}']= PPA_list[i].copy()
  del shape_list
  del PPA_list
  run_list.append(f'{config}')
  find_origin_config[f'{config}'] = config


def calculate_values(r):
    latency = total_latency_list[r]
    energy = total_energy_list[r]
    leakage = total_leakage_list[r]
    area = total_area_list[r]
    
    x_value = latency
    y_value = (energy / latency) + leakage
    z_value = area
    
    if args.search_accuracy == 1:
      m_value = total_similarity_list[r] 
    else:
      m_value = None
    return x_value, y_value, z_value, m_value

print("combine_CONFIG",len(combine_CONFIG))

NCPU=multiprocessing.cpu_count()

for i in range(0,len(combine_CONFIG),500000):
  with Pool(processes=NCPU*4) as pool:
      pool.map(init, combine_CONFIG[i:i+500000])
  print(f"{i}/{len(combine_CONFIG)} tasks completed.")
results_dict = Manager().dict()
lock = Manager().Lock() 
depth = 0
for exec_set in execute_set:
  print(exec_set)
  print(f"------------------{depth}-----------------")
  time.sleep(1)
  

  if os.path.isdir(f"{path}/CLUSTER/{args.heterogeneity}/{depth-2}"):
    shutil.rmtree(f"{path}/CLUSTER/{args.heterogeneity}/{depth-2}")

  if os.path.isdir(f"{path}/ELEMENT/{args.heterogeneity}/{depth-2}"):
    shutil.rmtree(f"{path}/ELEMENT/{args.heterogeneity}/{depth-2}")

 
  if os.path.isdir(f"{path}/BOOKSIM/{args.heterogeneity}/{depth-2}"):
    shutil.rmtree(f"{path}/BOOKSIM/{args.heterogeneity}/{depth-2}")
    
  exec_set_list_repeated = [[combine_CONFIG[i],exec_set] for i in range(len(combine_CONFIG))]
  if depth==0:
    start = time.time()
    with Pool(processes=NCPU-2) as pool:
      total_tasks = len(combine_CONFIG)
      completed_count = Value('i', 0)
      results = [] 
      for i in range(len(combine_CONFIG)):
        result = pool.apply_async(my_func, (exec_set_list_repeated[i],), callback=update_progress)
        results.append(result)
      output = [result.get() for result in results]
    print(f"pool_time", time.time()-start)
  else:
    with Pool(processes=NCPU-2) as pool:
      total_tasks = len(combine_CONFIG)
      completed_count = Value('i', 0)
      results = [] 
      for i in range(len(combine_CONFIG)):
        result = pool.apply_async(my_func, (exec_set_list_repeated[i],), callback=update_progress)
        results.append(result)
      output = [result.get() for result in results]

  end =time.time()
  print("time ",(end - start))
  alive_combine_CONFIG = []

  if depth > 0 :
    with ProcessPoolExecutor() as executor:
      results = list(executor.map(calculate_values, run_list))
    x, y, z, m = np.transpose(results) 
    labels = np.array(run_list)
 
    if args.search_accuracy == 1:
      paretoPoints = []
      point = [list(element) for element in zip(x,y,z,m)]
      
      for adc in adc_set:
        for cellbit in cellbit_set:
          selected_indices = [index for index, item in enumerate(labels) if f'{adc}, {cellbit}]' in item]
          filtered_point = [point[i] for i in selected_indices]
          if len(filtered_point) == 0:
            print("filtered_point",adc,cellbit,len(filtered_point))
            continue

          if args.heterogeneity == 1:
            paretoPoint = set(tuple(pt) for pt in filtered_point)
          else:
            paretoPoint = keep_efficient(filtered_point)
          paretoPoints.extend(paretoPoint)
          print("filtered_point",adc,cellbit,len(paretoPoint))
          print("paretoPoints",len(paretoPoints))
    else:
      point = [list(element) for element in zip(x,y,z)]
      if args.heterogeneity == 1:
        paretoPoints = set(tuple(pt) for pt in point)
      else:
        paretoPoints = keep_efficient(point)
      



    pareto_label={}
    if args.search_accuracy == 1:
      for i,e in enumerate(point):
        if str(e[0])+str(e[1])+str(e[2])+str(e[3]) in pareto_label.keys():
          pareto_label[str(e[0])+str(e[1])+str(e[2])+str(e[3])].append(labels[i])
        else:
          pareto_label[str(e[0])+str(e[1])+str(e[2])+str(e[3])]= [labels[i]]

    else:
      for i,e in enumerate(point):
        if str(e[0])+str(e[1])+str(e[2]) in pareto_label.keys():
          pareto_label[str(e[0])+str(e[1])+str(e[2])].append(labels[i])
        else:
          pareto_label[str(e[0])+str(e[1])+str(e[2])]= [labels[i]]
    if args.search_accuracy == 1:
      max_value = int(max(paretoPoints, key=lambda x: x[3])[3])

    paretoPoints_list=[]
    for pareto in paretoPoints:
      if args.search_accuracy == 1:
        tmp_pareto =[pareto[0],pareto[1],pareto[2],pareto[3]-max_value]
      else:
        tmp_pareto =[pareto[0],pareto[1],pareto[2]]
      paretoPoints_list.append(tmp_pareto)
      
    print("paretoPoints_list",len(paretoPoints_list))

    if args.search_accuracy == 1:
      w = [args.latency,args.power,args.area,args.accuracy]
      sign = np.array([False,False,False,False])
    else:
      w = [args.latency,args.power,args.area]
      sign = np.array([False,False,False])
    t = Topsis(paretoPoints_list, w, sign)
    t.calc()
    beam_list_score={}
    overlap_count = {} 
    run_list = Manager().list()
    print("rank_to_best_similarity",len(t.rank_to_best_similarity()))
    for beam in range(len(t.rank_to_best_similarity())):
      if len(overlap_count) < args.beam_size_m:
        if args.search_accuracy == 1:
          pal_set = pareto_label[str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][0])+str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][1])+str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][2])+str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][3]+ max_value)]
        else:
          pal_set = pareto_label[str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][0])+str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][1])+str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][2])]
        
        for pal in pal_set:
          if pal.split("_")[0] in overlap_count:
            if overlap_count[pal.split("_")[0]] < args.beam_size_n:
              alive_combine_CONFIG.append(pal)
              run_list.append(pal)
              beam_list_score[pal]=t.best_similarity[beam]
              overlap_count[pal.split("_")[0]] +=1
            else:
              continue
          else:
            alive_combine_CONFIG.append(pal)
            run_list.append(pal)
            beam_list_score[pal]=t.best_similarity[beam]
            overlap_count[pal.split("_")[0]] =1
      else:
        break
  
  else:
    for r in run_list:
      alive_combine_CONFIG.append(r)
  
  combine_CONFIG = alive_combine_CONFIG
  depth+=1

if not os.path.isdir(f"{navcim_dir}/Inference_pytorch/search_result/{args.model}_hetero_predict"):
  os.makedirs(f"{navcim_dir}/Inference_pytorch/search_result/{args.model}_hetero_predict")

if args.search_accuracy == 1:
  final_latency_f = open(f"./search_result/{args.model}_hetero_predict/final_result_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area},{args.accuracy}]_{args.heterogeneity}_{args.search_accuracy_metric}.txt",'w', newline='')
else:
  final_latency_f = open(f"./search_result/{args.model}_hetero_predict/final_result_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area}]_{args.heterogeneity}.txt",'w', newline='')
final_latency_wr = csv.writer(final_latency_f)
for final in combine_CONFIG:
  final_latency_wr.writerow([final,total_latency_list[final], (total_energy_list[final]/total_latency_list[final])+total_leakage_list[final],total_area_list[final],total_similarity_list[final]])
final_latency_f.close()

if args.search_accuracy == 1:
  final_log_f = open(f"./search_result/{args.model}_hetero_predict/final_log_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area},{args.accuracy}]_{args.heterogeneity}_{args.search_accuracy_metric}.txt",'w', newline='')
else:
  final_log_f = open(f"./search_result/{args.model}_hetero_predict/final_log_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area}]_{args.heterogeneity}.txt",'w', newline='')
final_log_wr = csv.writer(final_log_f)
for final_ in combine_CONFIG:
  num_chip1_sofar1,num_chip2_sofar1 = number_of_tile_sofar(mapping_info_list[final_])
  final_log_wr.writerow([final_,mapping_info_list[final_],num_chip1_sofar1,num_chip2_sofar1])
final_log_f.close()
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
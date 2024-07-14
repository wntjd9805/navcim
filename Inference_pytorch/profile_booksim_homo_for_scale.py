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
import torch.nn as nn
from torch.autograd import Variable
from utee import make_path
from utee import wage_util
from models import dataset
import torchvision.models as models
import shutil
import sys
import joblib

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
from sklearn.preprocessing import StandardScaler,MinMaxScaler, RobustScaler
import pandas as pd
import pickle

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--model', default='ResNet50', help='VGG16|ResNet50|NasNetA|LFFD')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 64)')
parser.add_argument('--SA_size_1_ROW', type=int, required=True, help='SA_size_1')
parser.add_argument('--SA_size_1_COL', type=int, required=True, help='SA_size_1')
parser.add_argument('--PE_size_1', type=int, required=True, help='PE_size_1')
parser.add_argument('--TL_size_1', type=int, required=True, help='TL_size_1')
parser.add_argument('--ADC', type=int, required=True, help='ADC')
parser.add_argument('--Cellbit', type=int, required=True, help='Cellbit')

args = parser.parse_args()
navcim_dir = os.getenv('NAVCIM_DIR')

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
# def MinMaxScaler(value):
#     newmax = 100
#     newmin = 0
#     max_ = 298256.8070576
#     min_ = 305.61844
#     rvalue = (value - min_)/(max_ - min_)*(newmax - newmin) + newmin

#     return rvalue
# def inverse_minmax(value):
#     newmax = 100
#     newmin = 0
#     max_ = 298256.8070576
#     min_ = 305.61844
#     return value*(max_ - min_)/(newmax-newmin) + min_
def inverse_minmax_latency(value,max,min):
    newmax = 100
    newmin = 0
    max_ = max
    min_ = min
    return value*(max_ - min_)/(newmax-newmin) + min_

def inverse_minmax_energy(value,max_,min_):
    newmax = 10
    newmin = 0

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

# class RegressionModel_energy(nn.Module):
#     def __init__(self):
#         super(RegressionModel_energy, self).__init__()
#         self.fc1 = nn.Linear(8,16)
#         self.bn1 = nn.BatchNorm1d(num_features = 16)
#         self.fc2 = nn.Linear(16,32)
#         self.bn2 = nn.BatchNorm1d(num_features = 32)
#         self.fc3 = nn.Linear(32,64)
#         self.bn3 = nn.BatchNorm1d(num_features = 64)
#         self.fc4 = nn.Linear(64, 128)
#         self.bn4 = nn.BatchNorm1d(num_features = 128)
#         self.fc5 = nn.Linear(128, 64)
#         self.bn5 = nn.BatchNorm1d(num_features = 64)
#         self.fc6 = nn.Linear(64, 32)
#         self.bn6 = nn.BatchNorm1d(num_features = 32)
#         self.fc7 = nn.Linear(32, 16)  
#         self.bn7 = nn.BatchNorm1d(num_features = 16)
#         self.fc8 = nn.Linear(16, 1)
#         # self.dropout = nn.Dropout(0.1)
#         self.relu = nn.ELU()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.fc4(x)
#         x = self.bn4(x)
#         x = self.relu(x)
#         x = self.fc5(x)
#         x = self.bn5(x)
#         x = self.relu(x)
#         x = self.fc6(x)
#         x = self.bn6(x)
#         x = self.relu(x)
#         x = self.fc7(x)
#         x = self.bn7(x)
#         x = self.relu(x)
#         x = self.fc8(x)
#         return x
    
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

#latency parameter
scaler_x_params_latency = joblib.load(f"{navcim_dir}/Inference_pytorch/predict_model/scaler_x_params_latency_noPGS.pkl")
scaler_y_params_latency = joblib.load(f"{navcim_dir}/Inference_pytorch/predict_model/scaler_y_params_latency_noPGS.pkl")

scaler_x_latency = StandardScaler()
scaler_x_latency.mean_ = scaler_x_params_latency['mean']
scaler_x_latency.scale_ = scaler_x_params_latency['scale']

# #Energy parameter
# scaler_x_params_energy = joblib.load(f"{navcim_dir}/Inference_pytorch/predict_model/scaler_x_params_energy_best_v2.pkl")
# scaler_y_params_energy = joblib.load(f"{navcim_dir}/Inference_pytorch/predict_model/scaler_y_params_energy_best_v2.pkl")

# # scaler_x_energy = StandardScaler()
# scaler_x_energy = RobustScaler()
# scaler_x_energy.center_ = scaler_x_params_energy['center']
# scaler_x_energy.scale_ = scaler_x_params_energy['scale']

#Power parameter
scaler_x_params_power = joblib.load(f"{navcim_dir}/Inference_pytorch/predict_model/scaler_x_params_power.pkl")
scaler_x_power = StandardScaler()
scaler_x_power.mean_ = scaler_x_params_power['mean']
scaler_x_power.scale_ = scaler_x_params_power['scale']



loaded_latency_model = RegressionModel_latency()
# loaded_energy_model = RegressionModel_energy()
loaded_power_model = RegressionModel_power()

device = torch.device("cpu")
# Load the saved model state dictionary
loaded_latency_model.load_state_dict(torch.load(f"{navcim_dir}/Inference_pytorch/predict_model/regression_model_latency_noPGS.pth",map_location=device))
# loaded_energy_model.load_state_dict(torch.load(f'{navcim_dir}/Inference_pytorch/predict_model/regression_model_energy_best_v2.pth',map_location=device))
loaded_power_model.load_state_dict(torch.load(f'{navcim_dir}/Inference_pytorch/predict_model/regression_model_power.pth',map_location=device))

# Set the model to evaluation mode
loaded_latency_model.eval()
# loaded_energy_model.eval()
loaded_power_model.eval()





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

import random
import math



##place and rounting
def profile_for_select(i ,key_loc, tmp_dfg, Nbit ,Bus_width, cluster_clk_freq, FPS ,CLUSTER1, iter, big_wr1,small_wr1, shape1, level, count, chip_width,chip_number,tile_grid,tile_type, dict_for_predict):
  row1 = int(shape1[2])
  col1 = int(shape1[3])
  num_of_row = math.ceil(row1/int(CLUSTER1))
  num_of_col = math.ceil(col1/int(CLUSTER1))
  number_of_cluster= math.ceil(int(shape1[4]) / (CLUSTER1**2))
  
    
  key_list=list(tmp_dfg.keys())
  numBitToLoadIn=None
  numBitToLoadOut=None
  numInVector = None

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

  if tmp_dfg[str(i)][0] =="nn.dense":
    numInVector = 1
    numBitToLoadIn = numInVector * Nbit * For_dense
  injection = injection_rate(numBitToLoadIn/num_of_row, Nbit, FPS, Bus_width, cluster_clk_freq)

  numBitToLoadOut = numInVector * Nbit * int(tmp_dfg[str(i)][3][1])
  tmp_dfg[str(i)][4] = numBitToLoadOut
  
  for r in range(number_of_cluster):
    big_wr1.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{iter[0]}-{iter[1]}",f"chip{chip_number}",numBitToLoadIn/num_of_row,injection,num_of_row,num_of_col])
    # zigzag
    tile_grid[iter[0],iter[1]] = i
    tile_type[iter[0],iter[1]] = chip_number
    count+=1    
    if (level+1)*(chip_width) <= count:
        iter[1]+=1
        level+=1
    else:
      if level%2==0:
        iter[0]+=1
      else:
        iter[0]-=1

  # print(mapping_info)
  dict_for_predict[str(i)]=[number_of_cluster, numBitToLoadIn/num_of_row, injection]  
  return iter, count, level, tile_grid, tile_type, num_of_col, number_of_cluster
  #----



def execute_booksim(node, cluster_width, chip1_width, cluster_flit_cycle, chip1_flit_cycle ,model,chip1, cluster_meter, chip1_meter ,chip1_buswidth, period):
  cmd1 = f'../booksim2/src/booksim ../booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {cluster_flit_cycle} {chip1_flit_cycle} ../Inference_pytorch/record_{args.model}/homo/CLUSTER/CLUSTER_{model}_{chip1}.txt ../Inference_pytorch/record_{args.model}/homo/ELEMENT/ELEMENT_{model}_{chip1}.txt {cluster_meter} {chip1_meter} {chip1_buswidth} {chip1_buswidth} 0 n/a {node} 1 | egrep "taken|Total Power|Total Area|Total leak Power" > ./record_{args.model}/homo/BOOKSIM/BOOKSIM_{model}_{chip1}.txt'
  print(cmd1)

  try:
    output = subprocess.check_output(
        cmd1, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as exc:
      sys.exit("Error!!!!!", exc.returncode, exc.output)
 
  fname = f"./record_{args.model}/homo/BOOKSIM/BOOKSIM_{model}_{chip1}.txt"
  latency_result = 0
  energy_result=0
  area_cluster=0
  cluster_leak_power = 0

  with open(fname) as f:
    lines = f.readlines()
    latency_result = int(lines[0].split("\n")[0].split(" ")[3])* period 
    energy_result = float(lines[1].split("\n")[0].split(" ")[3])* int(lines[0].split("\n")[0].split(" ")[3])* period
    area_cluster = float(lines[2].split("\n")[0].split(" ")[3])*1000000
    cluster_leak_power = float(lines[3].split("\n")[0].split(" ")[4])
  
  # print(node)
  # print(period)
  # print("latency_predict",int(lines[0].split("\n")[0].split(" ")[3])* period)
  # print("power_predict",float(lines[1].split("\n")[0].split(" ")[3]))
  return latency_result,energy_result*1000,area_cluster,cluster_leak_power,float(lines[1].split("\n")[0].split(" ")[3])

def predict_booksim(node, cluster_width, chip1_width, chip2_width, cluster_flit_cycle, chip1_flit_cycle, chip2_flit_cycle,model,chip1, cluster_meter, chip1_meter,chip2_meter ,chip1_buswidth,period ,num_of_input, num_of_dest, activation_size , injection_rate):

  latency_predict = 0
  energy_predict=0

  hop_average = calculate_average_hops(num_of_input, num_of_dest, (cluster_width,cluster_width))
  input_data = np.array([[cluster_width, cluster_flit_cycle, cluster_meter , injection_rate, activation_size,num_of_input, num_of_dest]])  
  input_data_power = np.array([[cluster_width, cluster_flit_cycle, cluster_meter*1000 , np.exp(injection_rate), np.log1p(activation_size),num_of_input, num_of_dest]])  
  input_data_energy = np.array([[cluster_width, cluster_meter , injection_rate, activation_size, num_of_input, num_of_dest, hop_average]])
  with torch.no_grad():
    # print(input_data)
    # X_normalized_latency = scaler_x_latency.transform(input_data)

    X_normalized_latency = scaler_x_latency.transform(input_data)
    latency_predict_normalized = loaded_latency_model(torch.tensor(X_normalized_latency,dtype=torch.float32))
    latency_predict = inverse_minmax_latency(latency_predict_normalized, scaler_y_params_latency['max'], scaler_y_params_latency['min'])
    
    X_normalized_power = scaler_x_power.transform(input_data)
    power_predict = loaded_power_model(torch.tensor(X_normalized_power,dtype=torch.float32))
    


    # input_data_energy[0][1] = input_data_energy[0][1] * 1000
    # input_data_energy[0][3] = np.log(input_data_energy[0][3])
    # input_data_energy = np.append(input_data_energy, latency_predict)
    # input_data_energy = input_data_energy.reshape(1,-1)

    # X_normalized_energy = scaler_x_energy.transform(input_data_energy)
    # energy_predict_normalized = loaded_energy_model(torch.tensor(X_normalized_energy,dtype=torch.float32))
    
    # latency_predict = scaler_y_latency.inverse_transform(latency_predict_normalized.reshape(-1,1))
    latency_predict = inverse_minmax_latency(latency_predict_normalized, scaler_y_params_latency['max'], scaler_y_params_latency['min'])
    # energy_predict = np.expm1(energy_predict_normalized)
    # energy_predict = np.expm1(inverse_minmax_energy(energy_predict_normalized, scaler_y_params_energy['max'], scaler_y_params_energy['min']))

  print(node)  
  # print("latency calculate: ",latency_predict_normalized*(scaler_y_latency.data_max_- scaler_y_latency.data_min_)+ scaler_y_latency.data_min_)
  # print("latency_predict",latency_predict)
  # print("energy_predict",energy_predict)

  latency_result = latency_predict
  # latency_result = latency_predict_normalized*(scaler_y_latency.data_max_- scaler_y_latency.data_min_)+ scaler_y_latency.data_min_* period 
 
  energy_result = energy_predict * 1000 
  # print(latency_result)
  # print(energy_result)
  # print(power_predict)
  return latency_result[0][0],0,power_predict[0][0]

def make_args(config):
  compute_latency = 0
  compute_power = 0
  NUM1_ROW = config[0][0]
  NUM1_COL = config[0][1]
  PE1 = config[0][2]
  Tile1 = config[0][3]
  ADC = config[0][4]
  Cellbit = config[0][5]
 
  shape1={}
  compute_PPA1={}
  node_col={}
  fname1 = f"./shape/shape_{args.model}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"

  with open(fname1) as f:
      lines = f.readlines()
      for i, l in enumerate(lines):
          l=l.replace("\n", "")
          shape1[i]=l.split(',')
  
  select_shape={}
  print(All_layer)
  for layer_num, layer in enumerate(All_layer):
    latency1 = LATENCY.get(f"{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")-inter_connect_LATENCY.get(f"{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
    energy1 = POWER.get(f"{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")-inter_connect_POWER.get(f"{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
    area1 = AREA.get(f"{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")-inter_connect_AREA.get(f"{layer}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
    compute_PPA1[layer_num] = [latency1,energy1,area1] 
    compute_latency = compute_latency + compute_PPA1[layer_num][0]
    compute_power = compute_power + energy1                              #??
    select_shape[layer_num] = shape1.get(layer_num)
  
  CLUSTER1 = 1
  tmp_dfg = dfg
  conv_dense = 0
  tot_cluster_tile=0
  FPS_latency = 0
  for i in tmp_dfg.keys():
    if tmp_dfg[str(i)][0]=='nn.conv2d' or tmp_dfg[str(i)][0]=='nn.dense':
      tmp_dfg[str(i)][5]=math.ceil(int(select_shape[conv_dense][4]) / (CLUSTER1**2))
      FPS_latency = FPS_latency + float(LATENCY.get(f"layer{int(conv_dense)+1}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"))*1e-9
      num_cluster = math.ceil(int(shape1[conv_dense][4]) / (CLUSTER1**2))
      tot_cluster_tile = tot_cluster_tile + num_cluster
      conv_dense=conv_dense+1

      
  
  #Placement&routing
  placed_coordinates = []
  result_list=[]
  Nbit=8
  Bus_width = 128
  #clkFreq = 1e9
  FPS = 1/FPS_latency
  conv_dense=0
  chip_width = math.ceil(math.sqrt(tot_cluster_tile))+1
  X = np.linspace(-1,-1,chip_width)
  Y = np.linspace(-1,-1,chip_width)
  tile_grid,tile_type = np.meshgrid(X,Y)

  chip1_flit_cycle = flit_cycle(unitLatencyRep.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"), unitLatencyWire.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"),tile_width_meter.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"),clk_frequency.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"),minDist.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"))
  # exit()
  chip2_flit_cycle = chip1_flit_cycle
  cluster_Rep = unitLatencyRep.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  cluster_wire = unitLatencyWire.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  cluster_meter = tile_width_meter.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")*CLUSTER1
  cluster_clk_freq = clk_frequency.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  cluster_flit_cycle = chip1_flit_cycle
  chip1_meter = tile_width_meter.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  chip2_meter = chip1_meter
  chip1_buswidth = busWidth.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  period = clk_period.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  # period = 1
  iter=[0,1]
  level = np.array([0], dtype=int)
  count = np.array([0], dtype=int)
  tmp_dfg = dfg


  total_latency = 0
  total_energy = 0
  booksim_total_latency = 0
  booksim_total_energy = 0
  compute_latency = 0
  total_energy = 0
  total_area = 0
  total_leakage =0
  chip1_area = chip1_meter**2*1e12
  chip1_leakage = leakage_POWER.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")/int(shape1[1][4])*1e-6
  chip1_booksim_area  = 0
  chip1_booksim_leakage  = 0
  
  big_f = open(f"./record_{args.model}/homo/CLUSTER/CLUSTER_{args.model}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}.txt",'w', newline='')
  big_wr = csv.writer(big_f)
  big_wr.writerow(["node","destination1","destination2","op","location","type","activation_size","injection_rate"])
  big_f.close()
  small_f = open(f"./record_{args.model}/homo/ELEMENT/ELEMENT_{args.model}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}.txt",'w', newline='')
  small_wr = csv.writer(small_f)
  small_wr.writerow(["node","used","activation_size","injection_rate"])
  small_f.close()

  dict_for_predict = {}
  df_scalefactor = pd.DataFrame({'booksim_latency':[],'booksim_power':[], 'booksim_leakage_store':[], 'pred_latency':[],'pred_energy':[],'pred_power':[],'neurosim_latency':[],'neurosim_energy':[]})
  data_for_scalefactor=[0 for i in range(8)]
  for key_loc,i in enumerate(tmp_dfg.keys()):
    if i == '0':
      big_f = open(f"./record_{args.model}/homo/CLUSTER/CLUSTER_{args.model}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}.txt",'a')
      big_wr = csv.writer(big_f)
      small_f = open(f"./record_{args.model}/homo/ELEMENT/ELEMENT_{args.model}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}.txt",'a')
      small_wr = csv.writer(small_f)
      num_of_col = None
      iter, count, level, tile_grid, tile_type,num_of_col, number_of_cluster = profile_for_select(i, key_loc, tmp_dfg , Nbit ,Bus_width, cluster_clk_freq, FPS ,CLUSTER1, iter, big_wr ,small_wr, select_shape.get(conv_dense), level, count, chip_width,1,tile_grid, tile_type, dict_for_predict)
      big_f.close()
      small_f.close()
      # result = float(subprocess.Popen(cmd,stdout=subprocess.PIPE, stdin=subprocess.PIPE,shell=True).stdout.readlines()[0].decode('utf-8').strip())
      booksim_latency ,booksim_energy,chip1_booksim_area,chip1_booksim_leakage,booksim_power  = execute_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,args.model,f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',cluster_meter,chip1_meter,chip1_buswidth,period)
      total_latency += booksim_latency + compute_PPA1[conv_dense][0]
      total_energy += booksim_energy +  compute_PPA1[conv_dense][1] 
      booksim_total_latency += booksim_latency 
      booksim_total_energy += booksim_energy
      
      node_col[i] = num_of_col
      conv_dense = conv_dense + 1
      
    elif (tmp_dfg[str(i)][0]=='nn.conv2d' or tmp_dfg[str(i)][0]=='nn.dense'):
      big_f_select1 = open(f"./record_{args.model}/homo/CLUSTER/CLUSTER_{args.model}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}.txt",'a')
      big_wr_select1 = csv.writer(big_f_select1)
      small_f_select1 = open(f"./record_{args.model}/homo/ELEMENT/ELEMENT_{args.model}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}.txt",'a')
      small_wr_select1 = csv.writer(small_f_select1)
      num_of_col1 = None
      iter1 = copy.deepcopy(iter)
      iter1, count1, level1, tile_grid1, tile_type1, num_of_col1, number_of_cluster= profile_for_select(i, key_loc, tmp_dfg , Nbit ,Bus_width, cluster_clk_freq, FPS ,CLUSTER1, iter1, big_wr_select1 ,small_wr_select1, shape1.get(conv_dense), level, count, chip_width, 1, tile_grid, tile_type, dict_for_predict)
      big_f_select1.close()
      small_f_select1.close()
      

      num_of_input = 0
      for key_iter_for_fing_dest,p in enumerate(tmp_dfg.keys()):
          if tmp_dfg[p][1]==i or tmp_dfg[p][2]==i:
            num_of_input += dict_for_predict[p][0]

      booksim_latency ,booksim_energy,chip1_booksim_area,chip1_booksim_leakage,booksim_power  = execute_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,args.model,f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',cluster_meter,chip1_meter,chip1_buswidth,period)
      booksim_latency_predict ,booksim_energy_predict ,booksim_power_predict= predict_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',cluster_meter,chip1_meter,chip2_meter,chip1_buswidth,period, num_of_input, dict_for_predict[str(i)][0],dict_for_predict[str(i)][1],dict_for_predict[str(i)][2]) 
      booksim_leakage_store = chip1_booksim_leakage
      #booksim의 latency, power 결과와 neurosim 결과, 순서:booksim,booksim,predict_latency,predict_power,neurosim_latency, neurosim_power 
      a = [booksim_latency, booksim_power, booksim_leakage_store ,booksim_latency_predict ,booksim_energy_predict, booksim_power_predict ,compute_PPA1[conv_dense][0], compute_PPA1[conv_dense][1] ]
      data_for_scalefactor = [data_for_scalefactor[i]+a[i] for i in range(8)]

      select_shape[conv_dense] = shape1.get(layer_num)
      tmp_dfg[str(i)][5]=math.ceil(int(select_shape[conv_dense][4]) / (CLUSTER1**2))
      
      total_latency += booksim_latency + compute_PPA1[conv_dense][0]
      total_energy += booksim_energy +  compute_PPA1[conv_dense][1] 
      booksim_total_latency += booksim_latency 
      booksim_total_energy += booksim_energy
      
      iter = copy.deepcopy(iter1)
      level = level1
      count = count1
      tile_grid = tile_grid1
      tile_type = tile_type1
     
      node_col[i] = num_of_col1
      conv_dense=conv_dense+1

    else:
      big_f = open(f"./record_{args.model}/homo/CLUSTER/CLUSTER_{args.model}_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}.txt",'a')
      big_wr = csv.writer(big_f)
      tmp_act_size=0
      tmp_num_before_layers=0
      key_loc_before_size=None
      for key_iter_for_fing_dest,p in enumerate(tmp_dfg.keys()):
          if tmp_dfg[p][1]==i or tmp_dfg[p][2]==i:
              key_loc_before_size = tmp_dfg[p][4]/node_col[p]
            
      injection = injection_rate(key_loc_before_size, Nbit, FPS, Bus_width, cluster_clk_freq)
    
      tile_grid[iter[0],iter[1]] = i
      tile_type[iter[0],iter[1]] = 3
      node_col[i]=1
      big_wr.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{iter[0]}-0","non_MAC",key_loc_before_size,injection,1,1])
      big_f.close()
      
      dict_for_predict[str(i)] = [1,key_loc_before_size,injection]
      num_of_input = 0
      for key_iter_for_fing_dest,p in enumerate(tmp_dfg.keys()):
          if tmp_dfg[p][1]==i or tmp_dfg[p][2]==i:
            num_of_input += dict_for_predict[p][0]

      booksim_latency ,booksim_energy,chip1_booksim_area,chip1_booksim_leakage,booksim_power  = execute_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,args.model,f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',cluster_meter,chip1_meter,chip1_buswidth,period)
      booksim_latency_predict ,booksim_energy_predict,booksim_power_predict  = predict_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',cluster_meter,chip1_meter,chip2_meter,chip1_buswidth,period, num_of_input, dict_for_predict[str(i)][0],dict_for_predict[str(i)][1],dict_for_predict[str(i)][2])
      
      a = [booksim_latency, booksim_power, 0, booksim_latency_predict ,booksim_energy_predict, booksim_power_predict ,compute_PPA1[conv_dense][0], compute_PPA1[conv_dense][1] ]
      
      data_for_scalefactor = [data_for_scalefactor[i]+a[i] for i in range(8)]
      df_to_append = pd.DataFrame([data_for_scalefactor], columns=df_scalefactor.columns)

      # Add the converted DataFrame to the existing df_scalefactor DataFrame.
      df_scalefactor = pd.concat([df_scalefactor, df_to_append], ignore_index=True)
      data_for_scalefactor=[0 for i in range(8)]
      
      total_latency += booksim_latency 
      total_energy += booksim_energy
      booksim_total_latency += booksim_latency 
      booksim_total_energy += booksim_energy
  
  if os.path.exists(f"booksim+neurosim_scalefactor_{args.model}_{ADC}_{Cellbit}_with_power.csv"):
    df_scalefactor.to_csv(f"{navcim_dir}/Inference_pytorch/meta_learner_dataset/booksim+neurosim_scalefactor_{args.model}_{ADC}_{Cellbit}_with_power.csv", mode='a', header=False)
  else:
    df_scalefactor.to_csv(f"{navcim_dir}/Inference_pytorch/meta_learner_dataset/booksim+neurosim_scalefactor_{args.model}_{ADC}_{Cellbit}_with_power.csv", mode='a', header=True)


  
  total_leakage = ((chip1_booksim_leakage/(chip_width**2))*(count+chip_width) + chip1_leakage) * total_latency * 1000
  total_area =  chip1_booksim_area + chip1_area*count
  final_latency_f = open(f"./search_result/{args.model}_homo/final_LATENCY_ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}.txt",'w', newline='')
  final_latency_wr = csv.writer(final_latency_f)
  print(total_leakage)
  print(total_energy)
  print(total_energy+total_leakage)
  print((total_energy+total_leakage)/total_latency)
  final_latency_wr.writerow([float(total_latency), float((total_energy+total_leakage)/total_latency), float(total_area)])

  print(booksim_total_latency ,booksim_total_energy/total_latency)
  print(([float(total_latency), float((total_energy+total_leakage)/total_latency), float(total_area)]))


 
###########main
if not os.path.exists(f"{navcim_dir}/Inference_pytorch/meta_learner_dataset/booksim+neurosim_scalefactor_{args.model}_{args.ADC}_{args.Cellbit}_with_power.csv"):
  if not os.path.exists(f"{navcim_dir}/Inference_pytorch/meta_learner_dataset"):
    os.makedirs(f"{navcim_dir}/Inference_pytorch/meta_learner_dataset")
  df_scalefactor = pd.DataFrame({'booksim_latency':[],'booksim_power':[], 'booksim_leakage_store':[], 'pred_latency':[],'pred_energy':[],'pred_power':[],'neurosim_latency':[],'neurosim_energy':[]})
  df_scalefactor.to_csv(f"{navcim_dir}/Inference_pytorch/meta_learner_dataset/booksim+neurosim_scalefactor_{args.model}_{args.ADC}_{args.Cellbit}_with_power.csv", mode='a')

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
All_layer = []

CONFIG = [[args.SA_size_1_ROW, args.SA_size_1_COL, args.PE_size_1, args.TL_size_1, args.ADC, args.Cellbit]]
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
        leakage_POWER[f'ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=leakagePower
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
print(graph_ir)
node={}
dfg={}


for i in parse_ir:
    # print(i)
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
            # print(input)
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

print("------------------------------------------------")
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
        if node[str(j)][0]=="nn.conv2d" :
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
        if node[str(j)][0]=="nn.conv2d" :
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



if not os.path.isdir(f"./record_{args.model}/homo/CLUSTER"):
  os.makedirs(f"./record_{args.model}/homo/CLUSTER")
if not os.path.isdir(f"./record_{args.model}/homo/BOOKSIM"):
  os.makedirs(f"./record_{args.model}/homo/BOOKSIM")
if not os.path.isdir(f"./record_{args.model}/homo/ELEMENT"):
  os.makedirs(f"./record_{args.model}/homo/ELEMENT")
if not os.path.isdir(f"./search_result/{args.model}_homo"):
  os.makedirs(f"./search_result/{args.model}_homo")
make_args(CONFIG)


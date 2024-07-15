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
from concurrent.futures import ProcessPoolExecutor
import itertools
import pandas as pd
from sys import path
from copy import deepcopy

from sys import path
navcim_dir = os.getenv('NAVCIM_DIR')
path.append(f"{navcim_dir}/TOPSIS-Python/")
path.append(f'{navcim_dir}/cross-sim/applications/dnn/inference')
import run_inference_for_search
from topsis import Topsis

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
# parser.add_argument('--sa',type=int ,required=True,help='test')
# parser.add_argument('--pe',type=int ,required=True,help='test')
# parser.add_argument('--tile',type=int ,required=True,help='test')
args = parser.parse_args()

def read_and_sort_file_config(file_path):
  with open(file_path, 'r') as file:
    lines = file.readlines()
    data = []
    check = []
    for line in lines:
      parts = line.split('"')[1].split('_')[0]
      if parts not in check:
        data.append(eval(parts))
        check.append(parts)
  return data

def injection_rate(activation_size, Nbit, FPS, bus_width, freq):
  rate = (activation_size * Nbit * FPS) / (bus_width * freq)
  return rate

def extract_numbers(s):
    numbers = re.findall(r'\d+', s)  
    return [int(num) for num in numbers]  
# 리스트를 4개씩 묶는 함수
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

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
  # print("data_in",data_in)
  # print("data_out",data_out)
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
  # print("distribute")
  # print(result)
  return result
         
   
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
  return iter, count, level, tile_grid, num_of_col ,addition_cluster
  #----

def profile_for_select_distribute(i , tmp_dfg, Nbit ,Bus_width, chip_clk_freq, cluster_clk_freq, FPS ,CLUSTER, iter, cluster_wr,element_wr, shape, level, count, chip_width,chip_number,tile_grid, mapping_info):
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
  
  return iter, count, level, tile_grid, num_of_col ,addition_cluster
    

  
def number_of_tile_sofar(mapping_info):
  num_chip1=0
  num_chip2=0
  # print(mapping_info)
  for t in mapping_info.values():
    # print(t)
    if t[1] == 0:
      num_chip1+=1
    elif t[1] == 1:
      num_chip2 +=1
  return num_chip1,num_chip2
  
def PPA_function(latency, energy, area):
  return latency+energy/latency+area

def execute_booksim(node, cluster_width, chip1_width, cluster_flit_cycle, chip1_flit_cycle,model,select, cluster_meter, chip1_meter, chip_period,cluster_period, cluster_buswidth ,chip1_buswidth, file_name, NUM_ROW_cluster, NUM_COL_cluster):
  cmd1 = f'{navcim_dir}/booksim2/src/booksim {navcim_dir}/booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {cluster_flit_cycle} {chip1_flit_cycle} {navcim_dir}/Inference_pytorch/record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{model}_{file_name}{select}.txt {navcim_dir}/Inference_pytorch/record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{model}_{file_name}{select}.txt {cluster_meter} {chip1_meter} {cluster_buswidth} {chip1_buswidth} 0 na {node} 1 | egrep "taken|Total Power|Total Area|Total leak Power" > ./record_{args.model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/BOOKSIM_{model}_{file_name}{select}{node}.txt'
  cmd2 = f'{navcim_dir}/booksim2/src/booksim {navcim_dir}/booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {cluster_flit_cycle} {chip1_flit_cycle} {navcim_dir}/Inference_pytorch/record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{model}_{file_name}{select}.txt {navcim_dir}/Inference_pytorch/record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{model}_{file_name}{select}.txt {cluster_meter} {chip1_meter} {cluster_buswidth} {chip1_buswidth} 0 na {node} 2 | egrep "taken|Total Power|Total Area|Total leak Power" >> ./record_{args.model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/BOOKSIM_{model}_{file_name}{select}{node}.txt'
  cmd3 = f'{navcim_dir}/booksim2/src/booksim {navcim_dir}/booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {cluster_flit_cycle} {chip1_flit_cycle} {navcim_dir}/Inference_pytorch/record_{model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{model}_{file_name}{select}.txt {navcim_dir}/Inference_pytorch/record_{model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{model}_{file_name}{select}.txt {cluster_meter} {chip1_meter} {cluster_buswidth} {chip1_buswidth} 0 na {node} 3 | egrep "taken|Total Power|Total Area|Total leak Power" >> ./record_{args.model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/BOOKSIM_{model}_{file_name}{select}{node}.txt'
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
  
  
  fname = f"./record_{args.model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/BOOKSIM_{model}_{file_name}{select}{node}.txt"
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
    except:
      print("Error!!!!!", lines, fname)
      raise Exception("Error!!!!!")
    energy_result = float(lines[1].split("\n")[0].split(" ")[3])* int(lines[0].split("\n")[0].split(" ")[3])* cluster_period+ float(lines[5].split("\n")[0].split(" ")[3])*int(lines[4].split("\n")[0].split(" ")[3])*chip_period  + float(lines[9].split("\n")[0].split(" ")[3])*int(lines[8].split("\n")[0].split(" ")[3])*chip_period
    area_cluster = float(lines[2].split("\n")[0].split(" ")[3])*1000000
    area_chip = float(lines[6].split("\n")[0].split(" ")[3])*1000000
    cluster_leak_power = float(lines[3].split("\n")[0].split(" ")[4])
    chip_leak_power = float(lines[7].split("\n")[0].split(" ")[4])

  # remove file
    
  # print(cluster_period)
  # print("energy_result",energy_result*1000)
  # print(area_cluster)
  # print(cluster_leak_power)
  # print(chip_leak_power)
  
  # print("CLUSTER: ", int(lines[0].split("\n")[0].split(" ")[3]) ,"cycles")
  # print("Send: ", int(lines[1].split("\n")[0].split(" ")[3]) ,"cycles")
  # print("Receive: ", int(lines[2].split("\n")[0].split(" ")[3]) ,"cycles")

  # print(node,cycle)
  # print(node,latency_result)
  # time.sleep(0.5)
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
    # print("simple_cull_end")
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
  # return sum([float(row[x]) <= float(candidateRow[x])*0.95 for x in range(len(row))]) == len(row)    

def keep_efficient(input_set):
    'returns Pareto efficient row subset of pts'
    # sort points by decreasing sum of coordinates
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
    # if args.distribute == 1:
    #   tail_distribute = "distribute"

  cluster_f = open(f"./record_{args.model}/CLUSTER/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}/CLUSTER_{args.model}_{file_name}{tail_distribute}.txt",'w', newline='')
  cluster_wr = csv.writer(cluster_f)
  cluster_wr.writerow(["node","destination1","destination2","op","location","type","activation_size","injection_rate"])
  cluster_f.close()

  element_f = open(f"./record_{args.model}/ELEMENT/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}/ELEMENT_{args.model}_{file_name}{tail_distribute}.txt",'w', newline='')
  element_wr = csv.writer(element_f)
  element_wr.writerow(["node","used","activation_size","injection_rate"])
  element_f.close()
  tot_cluster_tile = sum(num_tile.values())
  if args.distribute == 1:
    chip_width = math.ceil(math.sqrt(tot_cluster_tile)) + 1
  else:
    chip_width = math.ceil(math.sqrt(tot_cluster_tile)) + 1

  FPS = sum(FPS_tmp_dict.values())/len(config)
  # print("chip_width",chip_width)
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

  del num_tile
  del FPS_tmp_dict

  return shape_tmp_dict ,PPA_tmp_dict ,chip_width,FPS,iter,level,count,tile_grid,mapping_info,conv_dense,total_latency,total_energy,total_area,total_leakage,selected_list
  
def make_args(ht,config,config_cluster,tmp_dfg,shape1,compute_PPA1,chip_width,FPS,iter,level,count,tile_grid,mapping_info,conv_dense,total_latency,total_energy,total_area,total_leakage,total_similarity,dfg_key,selected_list,file_name):
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
  
  # flit_cycle(cluster_Rep, cluster_wire, cluster_meter, cluster_clk_freq)
  chip1_meter = tile_width_meter.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  chip1_buswidth = busWidth.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  cluster_buswidth = chip1_buswidth
  chip1_period = clk_period.get(f"ADC:{ADC}_Cellbit:{Cellbit}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  period = clk_period.get(f"ADC:{ADC_cluster}_Cellbit:{Cellbit_cluster}_SA_row:{NUM_ROW_cluster}_SA_col:{NUM_COL_cluster}_PE:{PE_cluster}_TL:{Tile_cluster}")
  


  # total_area = 0
  # total_leakage =0
  chip1_area = (chip1_meter**2)*1e12
  # print(f"layer{conv_dense}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
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
  # if args.distribute == 1:
  #    tail_distribute = tail_select+"distribute"

  Nbit=8
  node_col={}
  inverse = {}
  tail1=f"{tail_select}{ht}_"
  # if args.distribute == 1:
  #   tail1=f"{tail_select}{ht}_distribute"
    
  for i in dfg_key:
    # print(f"---------------------{i}---------------------")
    # print(tmp_dfg[str(i)][0])  
    if (tmp_dfg[str(i)][0]=='nn.conv2d' or tmp_dfg[str(i)][0]=='nn.dense'):
      os.makedirs(f"./record_{args.model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}", exist_ok=True)
      os.makedirs(f"./record_{args.model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}", exist_ok=True)
      os.makedirs(f"./record_{args.model}/BOOKSIM/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}", exist_ok=True)
      shutil.copy(f"./record_{args.model}/CLUSTER/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail_distribute}.txt" ,f"./record_{args.model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}.txt")
      shutil.copy(f"./record_{args.model}/ELEMENT/{args.heterogeneity}/{depth-1}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{args.model}_{file_name}{tail_distribute}.txt" ,f"./record_{args.model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{args.model}_{file_name}{tail1}.txt")
      iter1 = copy.deepcopy(iter)
      level1 = copy.deepcopy(level)
      count1 = copy.deepcopy(count)
      tile_grid1 = copy.deepcopy(tile_grid)
      mapping_info1 = copy.deepcopy(mapping_info)
      selected_list1= copy.deepcopy(selected_list)
      copy_lock = 1

      cluster_f_select1 = open(f"./record_{args.model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}.txt",'a')
      cluster_wr_select1 = csv.writer(cluster_f_select1)
      element_f_select1 = open(f"./record_{args.model}/ELEMENT/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/ELEMENT_{args.model}_{file_name}{tail1}.txt",'a')
      element_wr_select1 = csv.writer(element_f_select1)
      num_of_col1 = None
      if args.distribute == 1:
        iter1, count1, level1, tile_grid1,num_of_col1,addition_cluster= profile_for_select_distribute(i, tmp_dfg , Nbit ,chip1_buswidth, chip1_clk_freq, cluster_clk_freq, FPS ,CLUSTER1, iter1, cluster_wr_select1 ,element_wr_select1, shape1.get(conv_dense), level1, count1, chip_width, ht, tile_grid1,mapping_info1)
      else:
        iter1, count1, level1, tile_grid1,num_of_col1,addition_cluster= profile_for_select(i, tmp_dfg , Nbit ,chip1_buswidth, chip1_clk_freq, cluster_clk_freq, FPS ,CLUSTER1, iter1, cluster_wr_select1 ,element_wr_select1, shape1.get(conv_dense), level1, count1, chip_width, ht, tile_grid1,mapping_info1)
 
      cluster_f_select1.close()
      element_f_select1.close()
      # print(f'--------------layer{conv_dense}-----------------')
      # print(f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}')
      # print("compute latency", compute_PPA1[conv_dense])
      if depth<=1 :
        os.makedirs(f"./record_{args.model}/Prepared_data/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}", exist_ok=True)
        os.makedirs(f"./record_{args.model}/Prepared_data/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}", exist_ok=True)
        if os.path.isfile(f"./record_{args.model}/Prepared_data/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}{i}.txt"):
          fname = f"./record_{args.model}/Prepared_data/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}{i}.txt"
          try:
            with open(fname) as f:
              lines = f.readlines()
              booksim_latency1=float(lines[0].split(',')[0])
              booksim_energy1=float(lines[0].split(',')[1])
              cluster_booksim_area=float(lines[0].split(',')[2])
              chip1_booksim_area=float(lines[0].split(',')[3])
              cluster_booksim_leakage=float(lines[0].split(',')[4])
              chip1_booksim_leakage=float(lines[0].split(',')[5])
          except Exception as e:
            print(f"An error occurred: {str(e)}")
            booksim_latency1,booksim_energy1,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage = execute_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,args.model,tail1,cluster_meter,chip1_meter,chip1_period,period,cluster_buswidth,chip1_buswidth,file_name, NUM_ROW_cluster, NUM_COL_cluster)
            if booksim_latency1 >0 :
              prepared1 = open(f"./record_{args.model}/Prepared_data/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}{i}.txt",'w')
              wr_prepared1 = csv.writer(prepared1)
              wr_prepared1.writerow([booksim_latency1,booksim_energy1,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage])
            
        else:
          booksim_latency1,booksim_energy1,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage = execute_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,args.model,tail1,cluster_meter,chip1_meter,chip1_period,period,cluster_buswidth,chip1_buswidth,file_name, NUM_ROW_cluster, NUM_COL_cluster)
          if booksim_latency1 >0 :
            prepared1 = open(f"./record_{args.model}/Prepared_data/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}{i}.txt",'w')
            wr_prepared1 = csv.writer(prepared1)
            wr_prepared1.writerow([booksim_latency1,booksim_energy1,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage])
        
        # print([cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage])

      else:
        booksim_latency1,booksim_energy1,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage = execute_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,args.model,tail1,cluster_meter,chip1_meter,chip1_period,period,cluster_buswidth,chip1_buswidth,file_name, NUM_ROW_cluster, NUM_COL_cluster)

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

        if args.search_accuracy == 1:
          total_similarity1 = total_similarity + (Similarity_for_accuracy[f"{ADC},{Cellbit}"][f"{NUM1_ROW},{NUM1_COL}"][conv_dense-1]* math.pow(hessian_list[conv_dense-1],2))
          return iter1,level1,count1,tile_grid1,mapping_info1,conv_dense,total_latency1,total_energy1,total_area1,total_leakage1,selected_list1,total_similarity1
        else:
          return iter1,level1,count1,tile_grid1,mapping_info1,conv_dense,total_latency1,total_energy1,total_area1,total_leakage1,selected_list1,0

      conv_dense=conv_dense+1

    else:
      before_conv_result1 = inverse[i][0]
      # print(before_conv_result1)
      # print(before_conv_result2)
      key_loc_before_size=None
      before_node_list=[]
      for key_iter_for_fing_dest,p in enumerate(tmp_dfg.keys()):
          if tmp_dfg[p][1]==i or tmp_dfg[p][2]==i:
            key_loc_before_size = tmp_dfg[p][4]
            before_node_list.append(int(p))

      injection1 = injection_rate(key_loc_before_size/num_of_col1, Nbit, FPS,  chip1_buswidth, cluster_clk_freq)
      cluster_f_select1 = open(f"./record_{args.model}/CLUSTER/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}.txt",'a')
      cluster_wr_select1 = csv.writer(cluster_f_select1)
      cluster_wr_select1.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{iter1[0]}-0","non_MAC",key_loc_before_size/num_of_col1,injection1,1,1])
      cluster_f_select1.close()
      
 
      booksim_latency_non_mac = 0
      booksim_energy_non_mac = 0
      if depth<=1: 
        if os.path.isfile(f"./record_{args.model}/Prepared_data/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}{i}.txt"):
          fname = f"./record_{args.model}/Prepared_data/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}{i}.txt"
          try:
            with open(fname) as f:
              lines = f.readlines()
              booksim_latency_non_mac=float(lines[0].split(',')[0])
              booksim_energy_non_mac=float(lines[0].split(',')[1])
          except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("fname: ",fname)
            print("Lines content: ", lines)   
            booksim_latency_non_mac,booksim_energy_non_mac,cluster_booksim_area_non_mac,chip1_booksim_area_non_mac,cluster_booksim_leakage_non_mac,chip1_booksim_leakage_non_mac  = execute_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,args.model,tail1,cluster_meter,chip1_meter,chip1_period,period,cluster_buswidth,chip1_buswidth,file_name, NUM_ROW_cluster, NUM_COL_cluster)
            if booksim_latency_non_mac >0 :
              prepared1 = open(f"./record_{args.model}/Prepared_data/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}{i}.txt",'w')
              wr_prepared1 = csv.writer(prepared1)
              wr_prepared1.writerow([booksim_latency_non_mac,booksim_energy_non_mac])
        else:
          booksim_latency_non_mac,booksim_energy_non_mac,cluster_booksim_area_non_mac,chip1_booksim_area_non_mac,cluster_booksim_leakage_non_mac,chip1_booksim_leakage_non_mac  = execute_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,args.model,tail1,cluster_meter,chip1_meter,chip1_period,period,cluster_buswidth,chip1_buswidth,file_name, NUM_ROW_cluster, NUM_COL_cluster)
          if booksim_latency_non_mac >0 :
            prepared1 = open(f"./record_{args.model}/Prepared_data/{args.heterogeneity}/{depth}/{NUM_ROW_cluster}/{NUM_COL_cluster}/CLUSTER_{args.model}_{file_name}{tail1}{i}.txt",'w')
            wr_prepared1 = csv.writer(prepared1)
            wr_prepared1.writerow([booksim_latency_non_mac,booksim_energy_non_mac])
      else:
        booksim_latency_non_mac,booksim_energy_non_mac,cluster_booksim_area_non_mac,chip1_booksim_area_non_mac,cluster_booksim_leakage_non_mac,chip1_booksim_leakage_non_mac  = execute_booksim(i,chip_width,int(CLUSTER1),cluster_flit_cycle,chip1_flit_cycle,args.model,tail1,cluster_meter,chip1_meter,chip1_period,period,cluster_buswidth,chip1_buswidth,file_name, NUM_ROW_cluster, NUM_COL_cluster)

      # print("booksim_latency",inverse[i][1]+booksim_latency_non_mac)
      total_latency1 = total_latency + booksim_latency_non_mac + before_conv_result1[0]
      total_energy1 = total_energy + booksim_energy_non_mac + before_conv_result1[1]
      # print("booksim energy", inverse[i][1]+ booksim_energy_non_mac)
      if depth == 0 :
        # print(total_leakage, cluster_booksim_leakage, chip_width, chip1_booksim_leakage, addition_cluster, chip1_leakage)
        total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2))*chip_width + (cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
        total_area1 = total_area + cluster_booksim_area + (chip1_booksim_area+cluster_area)*addition_cluster
      else :
        total_leakage1 = total_leakage + ((cluster_booksim_leakage/(chip_width**2)+chip1_booksim_leakage)*addition_cluster + chip1_leakage)*1000
        total_area1 = total_area + (chip1_booksim_area+cluster_area)*addition_cluster
      
      selected_list1.append(ht)
      

      if args.search_accuracy == 1:
        total_similarity1 = total_similarity + (Similarity_for_accuracy[f"{ADC},{Cellbit}"][f"{NUM1_ROW},{NUM1_COL}"][conv_dense-1] * math.pow(hessian_list[conv_dense-1],2))
        return iter1,level1,count1,tile_grid1,mapping_info1,conv_dense,total_latency1,total_energy1,total_area1,total_leakage1,selected_list1,total_similarity1
      else:
        return iter1,level1,count1,tile_grid1,mapping_info1,conv_dense,total_latency1,total_energy1,total_area1,total_leakage1,selected_list1,0
      

def my_func(input):
    config = input[0]
    exec_set =input[1]
    config_origin = find_origin_config[f'{config}']
    file_name = make_file_name(config_origin)
    for ht in range(args.heterogeneity):
      iter1_tmp,level1_tmp,count1_tmp,tile_grid1_tmp,mapping_info1_tmp,conv_dense_tmp,total_latency1_tmp,total_energy1_tmp,total_area1_tmp,leakage1_tmp,selected_list1_tmp ,total_similarity_tmp= make_args(ht,config_origin[ht],config_origin[0], dfg, globals()['shape{}'.format(ht)][f'{config_origin}'] , globals()['compute_PPA{}_list'.format(ht)][f'{config_origin}'], chip_width_list[f'{config}'], FPS_list[f'{config}'], iter_list[f'{config}'], level_list[f'{config}'], count_list[f'{config}'], tile_grid_list[f'{config}'], mapping_info_list[f'{config}'], conv_dense_list[f'{config}'], total_latency_list[f'{config}'], total_energy_list[f'{config}'],total_area_list[f'{config}'],total_leakage_list[f'{config}'],total_similarity_list[f'{config}'],exec_set, all_selected_list[f'{config}'], file_name)
      with lock:
        selecte1 = f'_{ht}'
        chip_width_list[f'{config}{selecte1}'] = chip_width_list[f'{config}']
        FPS_list[f'{config}{selecte1}'] = FPS_list[f'{config}']
        iter_list[f'{config}{selecte1}']=iter1_tmp
        level_list[f'{config}{selecte1}']=level1_tmp
        count_list[f'{config}{selecte1}']=count1_tmp
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
All_layer = []

if args.search_accuracy == 1:
  file_path =f"{navcim_dir}/Inference_pytorch/search_result/{args.model}_hetero_predict/final_result_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area},{args.accuracy}]_{args.heterogeneity}_{args.search_accuracy_metric}.txt"

else:
  file_path =f"{navcim_dir}/Inference_pytorch/search_result/{args.model}_hetero_predict/final_result_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area}]_{args.heterogeneity}.txt"

combine_CONFIG = read_and_sort_file_config(file_path)
CONFIG = []
for configs in combine_CONFIG:
    for config in configs:
        if  config not in CONFIG:
            CONFIG.append(config)

ntest = 100
ntest_batch = 100
parameter_crosssim = {}
adc_cellbilt = []

for config in CONFIG:
  if [config[4],config[5]] not in adc_cellbilt:
    adc_cellbilt.append([config[4],config[5]])

Similarity_for_accuracy = {}
if args.search_accuracy == 1:
  #raise error
  for adc_cell in adc_cellbilt:
    adc = adc_cell[0]
    cellbit = adc_cell[1]
    df = pd.read_csv(f'{navcim_dir}/cross-sim/applications/dnn/inference/{args.model}_{args.search_accuracy_metric}_ADC:{adc}_CellBit{cellbit}_list.csv')
    Similarity_for_accuracy_tmp = {col: df[col].tolist() for col in df.columns}
    Similarity_for_accuracy[f"{adc},{cellbit}"] = Similarity_for_accuracy_tmp

    layerParams, sizes, onnx_model, quantize, adc_ranges_cadidate, index_row_col, dac_ranges, positiveInputsOnly, xy_pars, weight_for_quantized, config =  deepcopy(run_inference_for_search.init(args.model ,ntest, ntest_batch,cellbit,adc))
    parameter_crosssim[f"{args.model},{adc},{cellbit}"] = [layerParams, sizes, onnx_model, quantize, adc_ranges_cadidate, index_row_col, dac_ranges, positiveInputsOnly, xy_pars, weight_for_quantized,config]

  with open(f'{navcim_dir}/cross-sim/applications/dnn/inference/{args.model}_hessian_list.txt', 'r') as file:
    content = file.read()  # 파일의 내용을 읽음
  hessian_list = [float(number) for number in content.split(', ')]


# CONFIG = [[256, 256, 4, 8], [128, 128, 2, 6], [64, 256, 6, 18], [64, 256, 4, 16], [64, 128, 3, 9], [256, 128, 3, 6], [256, 256, 2, 6], [256, 256, 2, 4], [128, 128, 5, 10], [64, 128, 3, 6], [256, 128, 2, 6], [64, 64, 5, 15], [128, 256, 4, 12], [128, 64, 4, 8], [64, 128, 4, 8], [64, 64, 4, 16], [128, 256, 4, 8], [64, 128, 10, 20], [128, 64, 2, 10], [128, 64, 2, 6], [64, 128, 4, 20], [64, 128, 7, 14], [64, 64, 4, 8], [64, 128, 9, 18], [128, 128, 2, 8], [64, 64, 2, 6], [128, 256, 2, 8], [64, 64, 6, 12], [64, 64, 3, 18], [64, 256, 9, 18], [64, 64, 9, 18], [128, 128, 6, 12], [64, 128, 4, 12], [64, 128, 2, 8], [64, 128, 3, 12], [128, 128, 3, 9], [128, 64, 2, 12], [64, 128, 2, 12], [128, 128, 4, 12], [64, 256, 10, 20], [64, 128, 4, 16], [64, 128, 2, 6], [64, 64, 3, 12], [64, 64, 7, 14], [64, 128, 2, 18], [64, 64, 2, 4], [64, 256, 5, 10], [64, 128, 6, 12], [64, 64, 8, 16], [64, 256, 4, 12], [256, 128, 2, 4], [64, 256, 4, 8], [128, 256, 2, 4], [128, 256, 3, 9], [64, 128, 6, 18], [64, 64, 10, 20], [64, 64, 5, 20], [64, 64, 3, 15], [64, 128, 5, 10], [64, 64, 2, 12], [128, 128, 3, 12], [128, 64, 2, 8], [64, 64, 2, 10], [64, 128, 2, 14], [64, 128, 8, 16], [256, 256, 3, 6], [64, 128, 2, 10], [64, 128, 2, 20], [128, 128, 4, 8], [64, 64, 3, 9], [128, 64, 2, 4], [64, 64, 2, 20], [64, 128, 2, 16], [256, 256, 3, 9], [128, 256, 5, 10], [64, 64, 2, 18], [64, 64, 6, 18], [128, 256, 6, 12], [64, 256, 7, 14], [64, 128, 3, 18], [256, 256, 2, 8], [64, 64, 3, 6], [64, 64, 4, 12], [64, 256, 8, 16], [128, 64, 3, 6], [128, 256, 3, 6], [64, 128, 3, 15], [128, 256, 2, 10], [64, 128, 5, 15], [64, 64, 2, 8], [128, 128, 3, 6], [128, 64, 3, 9], [128, 128, 2, 12], [64, 64, 4, 20], [128, 256, 2, 6], [64, 64, 5, 10], [64, 128, 5, 20], [128, 64, 5, 10], [64, 256, 6, 12], [64, 256, 4, 20]]
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
# graph, params= tvm.relay.optimize(graph, target='llvm', params=params)
graph_ir = str(graph)
parse_ir = graph_ir.split('\n')
# print(graph_ir)
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
            if has_two == 1 and (args.model != "EfficientB0"):
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
            if has_two == 1 and (args.model != "EfficientB0"):
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




path = f"{navcim_dir}/Inference_pytorch/record_{args.model}"

 
print(dfg)

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
        if completed_count.value % 100 == 0:  
            print(f"{completed_count.value}/{total_tasks} tasks completed.")

def init(config):
  os.makedirs(f"{path}/CLUSTER/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}", exist_ok=True)
  os.makedirs(f"{path}/ELEMENT/{args.heterogeneity}/{depth}/{config[0][0]}/{config[0][1]}", exist_ok=True)
  shape_list, PPA_list ,chip_width_list[f'{config}'],FPS_list[f'{config}'],iter_list[f'{config}'],level_list[f'{config}'],count_list[f'{config}'],tile_grid_list[f'{config}'],mapping_info_list[f'{config}'],conv_dense_list[f'{config}'],total_latency_list[f'{config}'],total_energy_list[f'{config}'],total_area_list[f'{config}'],total_leakage_list[f'{config}'],all_selected_list[f'{config}'] = initialization(config)
  total_similarity_list[f'{config}'] = 0
  total_accuracy_list[f'{config}'] = 0
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
  with Pool(processes=NCPU-2) as pool:
    total_tasks = len(combine_CONFIG)
    completed_count = Value('i', 0)
    results = [] 
    for i in range(len(combine_CONFIG)):
      result = pool.apply_async(my_func, (exec_set_list_repeated[i],), callback=update_progress)
      results.append(result)
    output = [result.get() for result in results]
  print(f"end_pool")
  alive_combine_CONFIG = []
  if depth > 0 :
    start = time.time()
    with ProcessPoolExecutor() as executor:
      results = list(executor.map(calculate_values, run_list))
    x, y, z, m = np.transpose(results) 
    labels = np.array(run_list)
    if args.search_accuracy == 1:
      point = [list(element) for element in zip(x,y,z,m)]
    else:
      point = [list(element) for element in zip(x,y,z)]
    pareto_label={}
    if args.search_accuracy == 1:
      for i,e in enumerate(point):
        if str(e[0])+str(e[1])+str(e[2])+str(e[3]) in pareto_label.keys():
          pareto_label[str(e[0])+str(e[1])+str(e[2])+str(e[3])].append(labels[i])
          # print(pareto_label[str(e[0])+str(e[1])+str(e[2])+str(e[3])])
          # raise ValueError("same pareto")
        else:
          pareto_label[str(e[0])+str(e[1])+str(e[2])+str(e[3])]= [labels[i]]
    else:
      for i,e in enumerate(point):
        if str(e[0])+str(e[1])+str(e[2]) in pareto_label.keys():
          pareto_label[str(e[0])+str(e[1])+str(e[2])].append(labels[i])
          # print(pareto_label[str(e[0])+str(e[1])+str(e[2])])
          # raise ValueError("same pareto")
        else:
          pareto_label[str(e[0])+str(e[1])+str(e[2])]= [labels[i]]
    # paretoPoints = simple_cull_cython(point, args.search_accuracy)
    # paretoPoints = keep_efficient(point)
    paretoPoints = set(tuple(pt) for pt in point)

    if args.search_accuracy == 1:
      max_value = int(max(paretoPoints, key=lambda x: x[3])[3])

    # if len(paretoPoints) > args.beam_size_m:
    paretoPoints_list=[]


    for pareto in paretoPoints:
      if args.search_accuracy == 1:
        tmp_pareto =[pareto[0],pareto[1],pareto[2],pareto[3]-max_value]
      else:
        tmp_pareto =[pareto[0],pareto[1],pareto[2]]
      paretoPoints_list.append(tmp_pareto)
      
    if args.search_accuracy == 1:
      w = [args.latency,args.power,args.area,args.accuracy]
      sign = np.array([False,False,False,False])
    else:
      w = [args.latency,args.power,args.area]
      sign = np.array([False,False,False])
    t = Topsis(paretoPoints_list, w, sign)
    t.calc()
    # beam_list=[]
    beam_list_score={}
    overlap_count = {} 
    print(len(t.rank_to_best_similarity()))

    run_list = Manager().list()
    for beam in range(len(t.rank_to_best_similarity())):
      # print("beam",overlap_count.values())
      if len(overlap_count) < args.beam_size_m:
        if args.search_accuracy == 1:
          pal_set = pareto_label[str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][0])+str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][1])+str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][2])+str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][3]+ max_value)]
        else:
          pal_set = pareto_label[str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][0])+str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][1])+str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][2])]
        # print("pal.split()[0]",pal.split("_")[0])
        for pal in pal_set:
          # print(pal)
          if pal.split("_")[0] in overlap_count:
            # print("overlap",overlap_count[pal.split("_")[0]])
            if overlap_count[pal.split("_")[0]] < args.beam_size_n:
              alive_combine_CONFIG.append(pal)
              run_list.append(pal)
              beam_list_score[pal]=t.best_similarity[beam]
              overlap_count[pal.split("_")[0]] +=1
            else:
              continue
          else:
            # beam_list.append(paretoPoints_list[t.rank_to_best_similarity()[beam]-1])
            alive_combine_CONFIG.append(pal)
            run_list.append(pal)
            beam_list_score[pal]=t.best_similarity[beam]
            overlap_count[pal.split("_")[0]] =1

      else:
        break
    print("alive_cadiate_arch",len(overlap_count))
    print("alive_cadidate_all",len(alive_combine_CONFIG))
    # paretoPoints = beam_list
  
  else:
    for r in run_list:
      alive_combine_CONFIG.append(r)
  
  combine_CONFIG = alive_combine_CONFIG
  depth+=1

import ast

total_accuracy_list = Manager().dict()

if args.search_accuracy== 1:
  for final in combine_CONFIG:
      # final = "[[256, 128, 2, 4, 6, 4], [64, 64, 2, 12, 6, 4]]_1_1_1_1_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_0_0_0_0_0_0_0_0
      config_str = final.split("_")[0]
      arch_configs = ast.literal_eval(config_str)
      selection_str = final.split("_")[1:]
      adc_bit = arch_configs[0][-2]
      cell_bit = arch_configs[0][-1]
      layerParams, sizes, onnx_model, quantize, adc_ranges_cadidate, index_row_col, dac_ranges, positiveInputsOnly, xy_pars, weight_for_quantized, config = parameter_crosssim[f"{args.model},{adc_bit},{cell_bit}"] 
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
      total_accuracy_list[f'{final}'] = accuracy[0]*100

      # results[str(parallel[1])] = temp_list

if not os.path.isdir(f"{navcim_dir}/Inference_pytorch/search_result/{args.model}_hetero_validate"):
  os.makedirs(f"{navcim_dir}/Inference_pytorch/search_result/{args.model}_hetero_validate")

if args.search_accuracy == 1:
  final_latency_f = open(f"./search_result/{args.model}_hetero_validate/final_result_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area},{args.accuracy}]_{args.heterogeneity}_{args.search_accuracy_metric}.txt",'w', newline='')
else:
  final_latency_f = open(f"./search_result/{args.model}_hetero_validate/final_result_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area}]_{args.heterogeneity}.txt",'w', newline='')
final_latency_wr = csv.writer(final_latency_f)
for final in combine_CONFIG:
  final_latency_wr.writerow([final,total_latency_list[final], (total_energy_list[final]/total_latency_list[final])+total_leakage_list[final],total_area_list[final],total_accuracy_list[final]])
final_latency_f.close()

if args.search_accuracy == 1:
  final_log_f = open(f"./search_result/{args.model}_hetero_validate/final_log_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area},{args.accuracy}]_{args.heterogeneity}_{args.search_accuracy_metric}.txt",'w', newline='')
else:
  final_log_f = open(f"./search_result/{args.model}_hetero_validate/final_log_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area}]_{args.heterogeneity}.txt",'w', newline='')
final_log_wr = csv.writer(final_log_f)
for final_ in combine_CONFIG:
  num_chip1_sofar1,num_chip2_sofar1 = number_of_tile_sofar(mapping_info_list[final_])
  final_log_wr.writerow([final_,mapping_info_list[final_],num_chip1_sofar1,num_chip2_sofar1])
final_log_f.close()
  
  
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")




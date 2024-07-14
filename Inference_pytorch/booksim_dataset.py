import argparse
from audioop import ratecv
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from models import dataset
import torchvision.models as models
#from IPython import embed
from datetime import datetime
from subprocess import call

from parse import *
import math
import numpy as np
import subprocess
import random
import math
from multiprocessing import Pool,Manager
import multiprocessing 
import itertools
import pandas as pd
from sys import path


parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--network_size', type=int, default=1, help='network_size(n x n)')
parser.add_argument('--latency_per_flit',type=float ,default=0, help='distribute')
parser.add_argument('--tilewidth',type=float ,default=0, help='distribute')
parser.add_argument('--unitLatencyRep',type=float ,default=0, help='distribute')
parser.add_argument('--injection_rate',type=float ,default=0, help='distribute')

args = parser.parse_args()

navcim_dir = os.getenv('NAVCIM_DIR')

directory = "./training_dataset"
if not os.path.exists(directory):
    os.makedirs(directory)


def prior_random_sample():
    candidate2 = pd.read_csv(f'{navcim_dir}/Inference_pytorch/booksim_dataset_PGS.csv')
    cand = candidate2.values
    length = cand.shape[1]
    result = [0 for _ in range(length)]
    index = [i for i in range(length)]
    np.random.shuffle(index)
    for i in index:
        value = np.random.choice(cand[:, i])
        result[i] = value
        cand = cand[cand[:, i] == value]
    
    return result[0], result[1],result[2],result[3],result[4],result[5],result[6]

def execute_booksim(network_size, latency_per_flit, tile_width, inject_rate, activation_size, input_start_point, input_num, dest_start_point, dest_num):
  cmd1 = f"pueue add -- \'{navcim_dir}/booksim2/src/booksim {navcim_dir}/booksim2/src/examples/mesh88_lat {network_size} {latency_per_flit} {tile_width} {inject_rate} {activation_size} {input_start_point} {input_num} {dest_start_point} {dest_num} -1 | egrep \"taken|Total Power|Total Area|Total leak Power|Hops\" > ./training_dataset/{network_size}_{latency_per_flit}_{tile_width}_{inject_rate}_{activation_size}_{input_num}_{dest_num}.txt'"
  subprocess.Popen(cmd1,stdout=subprocess.PIPE, stdin=subprocess.PIPE,shell=True)
 
def injection_rate(activation_size, Nbit, FPS, bus_width, freq):
    rate = (activation_size * Nbit * FPS) / (bus_width * freq)
    return rate


def flit_cycle(unit_rep,unit_wire,tile_width,clk,minDist):
  numRepeater = math.ceil(tile_width/minDist)
  if numRepeater>0:
    return math.ceil((unit_rep) * tile_width * clk)
  else:
    return math.ceil((unit_wire) * tile_width * clk)

def activation_distribution_normalized(min_val, max_val):
    probabilities = [-0.07 * i + 2000000 for i in range(min_val, max_val + 1)]
    total = sum(probabilities)
    normalized_probabilities = [p / total for p in probabilities]
    return normalized_probabilities

def num_node_distribution_normalized(min_val, max_val):
    probabilities = [-0.7 * i + 20 for i in range(min_val, max_val + 1)]
    total = sum(probabilities)
    normalized_probabilities = [p / total for p in probabilities]
    return normalized_probabilities



# FPS=[20,40,60,80,100,120,140,160,180,200,220,240]
# TILEWIDTH = [0.001,0.005,0.0001,0.0005,0.00001,0.00005]
# UNIT_REP =[1e-06,1e-07,1e-08,5e-06,5e-07,5e-08]
# UNIT_WIRE = [1e-07,1e-08,1e-09,5e-07,5e-08,5e-09]
# NETWORK_SIZE = [2,3,4,5]
# ACTIVATION_SIZE = [10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,120000,140000,160000,180000,200000,250000,300000,350000,400000,450000,500000]
# NUM_INPUT = [1,2,3,4,5,6,7]
# NUM_DESTINATION = [1,2,3,4,5,6,7]
# count = 0
# for fps in FPS :
#   for tile_width in TILEWIDTH :
#     for unit_rep in UNIT_REP:
#       for unit_wire in UNIT_WIRE:
#         for network_size in NETWORK_SIZE:
#           for activation_size in ACTIVATION_SIZE:
#             for num_input in NUM_INPUT:
#               for num_destination in NUM_DESTINATION:
#                 if network_size**2 > num_input + num_destination :
#                   inject_rate = injection_rate(activation_size, Nbit, fps, bus_width, clock_freq)
#                   latency_per_flit = flit_cycle(unit_rep,unit_wire,tile_width,clock_freq,minDist)
#                   input_start_point = 0
#                   dest_start_point = num_input
#                   # while(1):
#                     # input_start_point = random.randrange(0,network_size**2 - num_input +1)
#                     # input_start_point = 0
#                     # dest_start_point = random.randrange(num_input -1 ,(network_size**2) - num_destination + 1)
#                     # print(input_start_point, dest_start_point,":",num_input,num_destination)
#                     # if (dest_start_point < input_start_point-num_destination) or (dest_start_point >= input_start_point+num_input):
#                       # break
#                   count = count +1
#                   print(count)
#                   while(1):
#                     result = subprocess.Popen("pueue | grep Running | wc -l", shell=True, stdout=subprocess.PIPE, encoding='utf-8')
#                     # print(int(result.stdout.readline().split()[0]))
#                     run_cnt = int(result.stdout.readline().split()[0])
#                     # print(run_cnt)
#                     if run_cnt < 78 :
#                       execute_booksim(network_size, latency_per_flit, tile_width, inject_rate, activation_size, input_start_point, num_input, dest_start_point, num_destination)
#                       break
                  

               
minDist=0.00010883
clock_period = (6.50252e-3)*20
clock_freq = (1/clock_period)*1e+9
Nbit=8
bus_width=128
count = 0

activation_probabilities = activation_distribution_normalized(1, 15000000)
num_node_probabilities = num_node_distribution_normalized(1, 20)
while(1):  
  inject_rate = random.uniform(0.001, 0.3)
  tile_width = random.uniform(0.0001, 0.01)
  unit_rep = random.choice([3.77283e-07, 3.54362e-07])
  unit_wire = random.choice([3.7496e-08, 9.37399e-09])
  network_size = random.randint(2, 17)
  activation_size = np.random.choice(range(1, 15000000 + 1), p=activation_probabilities)
  if random.choice([True, False]):     
    num_input = 1    
    num_destination = np.random.choice(range(1, 20 + 1), p=num_node_probabilities)
  else:     
    num_input = np.random.choice(range(1, 20 + 1), p=num_node_probabilities)    
    num_destination = 1
  

  if network_size**2 > num_input + num_destination :
    # inject_rate = injection_rate(activation_size, Nbit, fps, bus_width, clock_freq)
    latency_per_flit = flit_cycle(unit_rep,unit_wire,tile_width,clock_freq,minDist)
    
    input_start_point = 0
    dest_start_point = num_input
    # while(1):
      # input_start_point = random.randrange(0,network_size**2 - num_input +1)
      # input_start_point = 0
      # dest_start_point = random.randrange(num_input -1 ,(network_size**2) - num_destination + 1)
      # print(input_start_point, dest_start_point,":",num_input,num_destination)
      # if (dest_start_point < input_start_point-num_destination) or (dest_start_point >= input_start_point+num_input):
        # break
    count = count +1
    print(count)
    while(1):
      result = subprocess.Popen("pueue | grep Queued  | wc -l", shell=True, stdout=subprocess.PIPE, encoding='utf-8')
      # print(int(result.stdout.readline().split()[0]))
      run_cnt = int(result.stdout.readline().split()[0])
      # print(run_cnt)
      if run_cnt < 10 :
        # network_size, latency_per_flit, tile_width, inject_rate, activation_size, num_input, num_destination = prior_random_sample()
        execute_booksim(network_size, latency_per_flit, tile_width, inject_rate, activation_size, input_start_point, num_input, dest_start_point, num_destination)
        break



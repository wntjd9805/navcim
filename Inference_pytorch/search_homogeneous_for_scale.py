import argparse
import os
import multiprocessing 
from itertools import combinations
from subprocess import Popen
import time
import subprocess
import random
import re
import copy
import signal
import numpy as np
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--model', default='ResNet50', help='VGG16|ResNet50|NasNetA|LFFD')
args = parser.parse_args()

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
  
print(len(CONFIG))


for config in CONFIG:
    print(config)
    process = Popen(f"pueue add -g {args.model} python profile_booksim_homo_for_scale.py --model={args.model} --SA_size_1_ROW={config[0]} --SA_size_1_COL={config[1]} --PE_size_1={config[2]} --TL_size_1={config[3]} --ADC={config[4]} --Cellbit={config[5]}", stdout=subprocess.PIPE, shell=True)
    time.sleep(0.1)

    
print("finish!!!!!")



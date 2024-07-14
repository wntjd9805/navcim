import re
import argparse
from itertools import combinations

parser = argparse.ArgumentParser(description='Extract TOP-K')

parser.add_argument('--models', required=True, help='VGG16|ResNet50|NasNetA|LFFD')
parser.add_argument('--heterogeneity', type=int, default=2, help='heterogeneity')
parser.add_argument('--comb_heterogeneity', type=int, default=2, help='heterogeneity')
parser.add_argument('--distribute',type=int ,default=1, help='distribute')
parser.add_argument('--beam_size_m',type=int ,default=700,help='beam_size_m')
parser.add_argument('--beam_size_n',type=int ,default=3,help='beam_size_n')
parser.add_argument('--weights',type=str ,required=True,help='consist of [latency,power,area] with pareto weight')
parser.add_argument('--accuracy',type=int ,default=1,help='weight_accuracy_with_pareto')
parser.add_argument('--search_accuracy',type=int ,default=0, help='search_accuracy')
parser.add_argument('--topk',type=int ,default=20,help='how many configs to extract')
args = parser.parse_args()

import os
navcim_dir = os.getenv('NAVCIM_DIR')

numbers = re.findall(r'\d+', args.weights)
numbers = list(map(int, numbers))
model_weight = [numbers[i:i+3] for i in range(0, len(numbers), 3)]

model_list = args.models.split(',')
pattern = r"\[[\d+, ]+\] *"
model_hap=[]
for i,model in enumerate(model_list):
    model_1 = []
    if args.search_accuracy == 1:
        path_simul = f'{navcim_dir}/Inference_pytorch/search_result/{model}_hetero_predict/final_result_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{model_weight[i][0]},{model_weight[i][1]},{model_weight[i][2]},{args.accuracy}]_{args.heterogeneity}_cka.txt'
    else:
        path_simul = f'{navcim_dir}/Inference_pytorch/search_result/{model}_hetero_predict/final_result_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{model_weight[i][0]},{model_weight[i][1]},{model_weight[i][2]}]_{args.heterogeneity}.txt'
    with open(path_simul, 'r') as files:
        for line in files.readlines():
            extracted_part = re.findall(pattern, line)
            if extracted_part:
                for arch in extracted_part:
                    if eval(arch) not in model_1:
                        model_1.append(eval(arch))
    
    model_hap.append(model_1[:])
    print(len(model_1))

hap=[]
count = [0 for i in model_list]

stop = True
while sum(count) < (args.topk * len(count)) and stop:
    for i in range(len(model_list)):    
        if len(model_hap[i]) > count[i]:
            if model_hap[i][count[i]] not in hap:
                hap.append(model_hap[i][count[i]])
                count[i] += 1
            else:
                model_hap[i].pop(count[i])
        else:
            stop = False

comb= list(combinations(hap,args.comb_heterogeneity))

delete_id = []
for i in range(len(comb)):
    check = None
    for c in comb[i]:
        if check == None:
            check = c[4:]
        else:
            if check != c[4:]:
                delete_id.append(i)
                break


for i in range(len(delete_id)):
    comb.pop(delete_id[i]-i)

#check duplicate
# configurations_set = set(tuple(map(tuple, config)) for config in comb)
# if len(comb) == len(configurations_set):
#     print("No duplicates")
# else:
#     print("Duplicate")

if args.search_accuracy == 1:
    with open(f'model_search_result_architechture_{args.models}_hetero_with_accuracy.txt','w') as file:
        for data in comb:
            file.write(str(list(data))+':')
else:
    with open(f'model_search_result_architechture_{args.models}_hetero.txt','w') as file:
        for data in comb:
            file.write(str(list(data))+':')
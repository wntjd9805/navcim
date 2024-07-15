import os
import argparse
import re
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--model', required=True)
parser.add_argument('--date', default="default")
parser.add_argument('--search_accuracy',type=int, required=True)
parser.add_argument('--heterogeneity', required=True)
parser.add_argument('--type', required=True)
# parser.add_argument('--sa',type=int ,required=True,help='test')
# parser.add_argument('--pe',type=int ,required=True,help='test')
# parser.add_argument('--tile',type=int ,required=True,help='test')
args = parser.parse_args()

# 설정값 정의
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

def parse_cpp_file(file_path):
    params = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    param_pattern = re.compile(r'(\w+)\s*=\s*([^;]+);')
    description_pattern = re.compile(r'//\s*(.*)')

    for line in lines:
        param_match = param_pattern.search(line)
        if param_match:
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            description = ""
            description_match = description_pattern.search(line)
            if description_match:
                description = description_match.group(1).strip()
            params[param_name] = {
                "value": param_value,
                "description": description
            }
    
    return params


def save_neurosim_params_to_file(params, output_path):
    with open(output_path, 'w') as file:
        for param_name, details in params.items():
            value = details['value']
            
            if param_name == "operationmode":
                if value == "1":
                    file.write("operationmode : conventionalSequential (Use several multi-bit RRAM as one synapse)\n")
                elif value == "2":
                    file.write("operationmode : conventionalParallel (Use several multi-bit RRAM as one synapse)\n")

            elif param_name == "memcelltype":
                if value == "1":
                    file.write("memcelltype : cell.memCellType = Type::SRAM\n")
                elif value == "2":
                    file.write("memcelltype : cell.memCellType = Type::RRAM\n")
                elif value == "3":
                    file.write("memcelltype : cell.memCellType = Type::FeFET\n")

            elif param_name == "accesstype":
                if value == "1":
                    file.write("accesstype : cell.accessType = CMOS_access\n")
                elif value == "2":
                    file.write("accesstype : cell.accessType = BJT_access\n")
                elif value == "3":
                    file.write("accesstype : cell.accessType = diode_access\n")
                elif value == "4":
                    file.write("accesstype : cell.accessType = none_access (Crossbar Array)\n")

            elif param_name == "transistortype":
                if value == "1":
                    file.write("transistortype : inputParameter.transistorType = conventional\n")

            elif param_name == "deviceroadmap":
                if value == "1":
                    file.write("deviceroadmap : inputParameter.deviceRoadmap = HP\n")
                elif value == "2":
                    file.write("deviceroadmap : inputParameter.deviceRoadmap = LSTP\n")
            
            elif param_name == "globalBufferType":
                if value == "false":
                    file.write("globalBufferType : register file\n")
                elif value == "true":
                    file.write("globalBufferType : SRAM\n")
            
            elif param_name == "tileBufferType":
                if value == "false":
                    file.write("tileBufferType : register file\n")
                elif value == "true":
                    file.write("tileBufferType : SRAM\n")
            
            elif param_name == "peBufferType":
                if value == "false":
                    file.write("peBufferType : register file\n")
                elif value == "true":
                    file.write("peBufferType : SRAM\n")
            
            elif param_name == "chipActivation":
                if value == "false":
                    file.write("chipActivation : activation (reLu/sigmoid) inside Tile\n")
                elif value == "true":
                    file.write("chipActivation : activation outside Tile\n")
            
            elif param_name == "reLu":
                if value == "false":
                    file.write("reLu : sigmoid\n")
                elif value == "true":
                    file.write("reLu : reLu\n")
            
            elif param_name == "novelMapping":
                if value == "false":
                    file.write("novelMapping : false\n")
                elif value == "true":
                    file.write("novelMapping : true\n")
            
            elif param_name == "SARADC":
                if value == "false":
                    file.write("SARADC : MLSA\n")
                elif value == "true":
                    file.write("SARADC : sar ADC\n")
            
            elif param_name == "currentMode":
                if value == "false":
                    file.write("currentMode : MLSA use VSA\n")
                elif value == "true":
                    file.write("currentMode : MLSA use CSA\n")
            
            elif param_name == "pipeline":
                if value == "false":
                    file.write("pipeline : layer-by-layer process --> huge leakage energy in HP\n")
                elif value == "true":
                    file.write("pipeline : pipeline process\n")
            
            elif param_name == "validated":
                if value == "false":
                    file.write("validated : no calibration factors\n")
                elif value == "true":
                    file.write("validated : validated by silicon data (wiring area in layout, gate switching activity, post-layout performance drop...)\n")
            
            elif param_name == "synchronous":
                if value == "false":
                    file.write("synchronous : asynchronous\n")
                elif value == "true":
                    file.write("synchronous : synchronous, clkFreq will be decided by sensing delay\n")
            
            elif param_name in ["globalBufferCoreSizeRow", "globalBufferCoreSizeCol", "tileBufferCoreSizeRow", "tileBufferCoreSizeCol", "speedUpDegree", "algoWeightMax", "algoWeightMin", "clkFreq", "temp", "technode"]:
                file.write(f"{param_name} = {value}\n")
def save_booksim_params_to_file(input_file_path, output_file_path,exclude_keywords):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()
    
    with open(output_file_path, 'w') as output_file:
        for line in lines:
            stripped_line = line.lstrip()
            if not stripped_line.startswith('//') and not any(keyword in stripped_line for keyword in exclude_keywords):
                output_file.write(stripped_line)


def remove_comments(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()
    
    with open(output_file_path, 'w') as output_file:
        for line in lines:
            stripped_line = line.lstrip()
            if not stripped_line.startswith('//'):
                output_file.write(stripped_line)

# 폴더 생성 함수 정의
if args.type =="folder":
    base_path = "."
    if args.search_accuracy == 0:
        folder_name = f"NavCim_log/{args.model}/accuracy_false/ADC_{adc_set}/CellBit_{cellbit_set}/Tile_{tile_set}/PE_{pe_set}/SA_{sa_set}/heterogeneity_{args.heterogeneity}/{args.date}"
    else:   
        folder_name = f"NavCim_log/{args.model}/accuracy_true/Tile_{tile_set}/PE_{pe_set}/SA_{sa_set}/ADC_{adc_set}/CellBit_{cellbit_set}/heterogeneity_{args.heterogeneity}/{args.date}"

    path = os.path.join(base_path, folder_name)
    os.makedirs(path, exist_ok=True)


elif args.type =="neurosim":
    if args.search_accuracy == 0:
        output_file_path = f"./NavCim_log/{args.model}/accuracy_false/ADC_{adc_set}/CellBit_{cellbit_set}/Tile_{tile_set}/PE_{pe_set}/SA_{sa_set}/heterogeneity_{args.heterogeneity}/{args.date}/NeuroSim_parameter.txt"
    else:   
        output_file_path = f"./NavCim_log/{args.model}/accuracy_true/Tile_{tile_set}/PE_{pe_set}/SA_{sa_set}/ADC_{adc_set}/CellBit_{cellbit_set}/heterogeneity_{args.heterogeneity}/{args.date}/NeuroSim_parameter.txt"
   
    params = parse_cpp_file('./NeuroSIM/Param.cpp')
    save_neurosim_params_to_file(params, output_file_path)

elif args.type == "booksim":
    if args.search_accuracy == 0:
        output_file_path = f"./NavCim_log/{args.model}/accuracy_false/ADC_{adc_set}/CellBit_{cellbit_set}/Tile_{tile_set}/PE_{pe_set}/SA_{sa_set}/heterogeneity_{args.heterogeneity}/{args.date}/BookSim_parameter.txt"
    else:   
        output_file_path = f"./NavCim_log/{args.model}/accuracy_true/Tile_{tile_set}/PE_{pe_set}/SA_{sa_set}/ADC_{adc_set}/CellBit_{cellbit_set}/heterogeneity_{args.heterogeneity}/{args.date}/BookSim_parameter.txt"
    exclude_keywords = ['k', 'latency_per_flit', 'wire_length_tile', 'injection_rate', 'flit_size']
    save_booksim_params_to_file('../booksim2/src/examples/mesh88_lat', output_file_path, exclude_keywords)
    

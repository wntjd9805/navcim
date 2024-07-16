# NavCIM
NavCIM is an end-to-end design automation tool for neuromorphic architectures by integrating heterogeneous tile/PE/SA sizes and ADC/Cellbit. For full details, please see our recent PACT 2024 paper.
If you use this tool in your research, please cite : {}
## Pre-requisites

### System dependencies

We tested our code on Ubuntu 18.04 amd64 system. We recommend using the Nvidia docker image.

```jsx
docker pull nvidia/cuda:11.0.3-devel-ubuntu18.04
```

### Software dependencies

Software pre-requisites for installing from the source should be satisfied for the following repositories:

- [DNN_NeuroSim_V1.3](https://github.com/neurosim/DNN_NeuroSim_V1.3.git)
- [Booksim2](https://github.com/booksim/booksim2.git)
- [TVM](https://github.com/apache/tvm)
- [crosssim](https://github.com/sandialabs/cross-sim)

We extended NeuroSim 1.3, BookSim2 and CrossSim2 to simulate the ReRAM-based analog CiM architecture with a mesh interconnect. Please the follow this documentation to install all dependencies

In Ubuntu 18.04 amd64 system, following commands install package dependencies:

```jsx
apt-get update
apt-get install wget gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev git python3-pip aria2 flex bison curl unzip
```

Install CMake (>= 3.18):

```bash
aria2c -q -d /tmp -o cmake-3.21.0-linux-x86_64.tar.gz  https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz
tar -zxf /tmp/cmake-3.21.0-linux-x86_64.tar.gz --strip=1 -C /usr

```

Install Clang and LLVM

```jsx
wget -c https://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar -xvf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
cp -rl clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04/* /usr/local
rm -rf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04 \
       clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
```

Install rust and pueue

```jsx
curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh
cargo install pueue
```

## Setup

Install and build navcim repositories from the source. We prepared the installation script (./install.sh):

```bash
cd "$HOME"
git clone --recursive https://github.com/wntjd9805/navcim.git navcim
cd navcim
git submodule update --init --recursive

mkdir $HOME/navcim/tvm/build
cp $HOME/navcim/config.cmake $HOME/navcim/tvm/build
cd $HOME/navcim/tvm/build
cmake ..
make -j 8

cd $HOME/navcim/booksim2/src
make -j 8

cd $HOME/navcim/Inference_pytorch/NeuroSIM
make -j 8
cd "$HOME"
```

Next, install Python (== 3.6, 3.10) dependencies.

Note: you need to use specific version of the library. See requirements_neurosim.txt, requirements_navcim.txt

```jsx
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/etc/profile.d/conda.sh

conda create -n neurosim python=3.6.9
conda activate neurosim
pip install -r $HOME/navcim/requirments_neurosim.txt

conda create -n navcim python=3.10
conda activate navcim
pip install -r $HOME/navcim/requirments_navcim.txt
```

Nex, install library for cross-sim and download ImageNet

```jsx
conda activate navcim
cd /$HOME/navcim/cross-sim
pip install .
mkdir -p /$HOME/navcim/cross-sim/applications/dnn/data/datasets/
cd /$HOME/navcim/cross-sim/applications/dnn/data/datasets/
wget -O imagenet.zip https://www.dropbox.com/scl/fi/nswi46sa5hti0dhxhyzuu/imagenet.zip?rlkey=2bue10y0v1nq0gwmfiqzoou2j&st=x6wd7wmn&dl=1 
```

Finally, you need to set the following environment variables and include them to .bashrc for later session.

```bash
export TVM_HOME=/$HOME/navcim/tvm 
export PYTHONPATH=/$HOME/navcim/tvm/python
export NAVCIM_DIR=/$HOME/navcim
```

## Run

### Preparation before searching

Sets up the search space necessary for the experimental stage. 
It includes options for whether accuracy is considered in the search (w/_accuracy) or not(w/o_accuracy) 
```bash
bash script_define_search_space.sh {option}
option = [w/_accuracy,w/o_accuracy]

# For example, If you don't want to consider accuracy, enter
# bash script_define_search_space.sh w/o_accuracy
```

Make weight file and execute Neurosim
```bash
bash run_neurosim.sh {model_name}
# model_name : **['ResNet50','EfficientNetB0','MobileNetV2','SqueezeNet']**
# For example, to run the MobileNetV2 model, enter
# bash run_neurosim.sh MobileNetV2
```

Training the Meta Learner
```bash
bash script_meta_learner.sh {model_name}
# model_name : **['ResNet50','EfficientNetB0','MobileNetV2','SqueezeNet']**
# For example, If you want to train MobileNetV2's meta-learner, enter
# bash script_meta_learner.sh MobileNetV2
```

If you want to do an accuracy aware search, you need to do this step. Otherwise, you can skip it.
```bash
bash make_cka.sh {model_name}
python hessian.py |model={model_name}
# model_name : **['ResNet50','EfficientNetB0','MobileNetV2','SqueezeNet']**
# For example
# bash make_cka.sh MobileNetV2
# python hessian.py |model=MobileNetV2
```

### Single Model Search

Here are the parameters you can set for the Single Model Search:

- **`model_name`**: Specify the name of the model you want to run.

- **Weights for TOPSIS Evaluation**
  - **`latency_weight`**: Adjust this weight to prioritize latency during the model selection process.
  - **`power_weight`**: Adjust this weight to prioritize power consumption during the model selection process.
  - **`area_weight`**: Adjust this weight to prioritize area during the model selection process.
  - These weights help customize the criteria for selecting the optimal model based on the TOPSIS evaluation method.

- **`heterogeneity`**: Specify the number of tiles to use during the search. Using multiple tiles enhances the diversity and breadth of the search.

- **`accuracy_aware`**:
  - Decide whether to include accuracy in the search criteria:
    - `0`: Exclude accuracy from the search.
    - `1`: Include accuracy in the search.

- **`guide_strategy`**:
  - Choose from two guide strategies to direct the search process:
    - **`constrain guide search`**:
      - Type `constrain` to proceed with exploring under specific constraints
      - (e.g., homogeneous architecture search results).
    - **`weight guide search`**:
      - Enter weights in the format `weight[latency, power, area]`, like `weight[2,1,1]`. This adjusts the importance of each performance metric to guide the search.

```bash
pueued -d #launch pueue to use multiple cores
bash script_single_model.sh {model_name} {latency_weight} {power_weight} {area_weight} {heterogeneity} {accuracy_aware} {guide_strategy}

# For example
# bash script_single_model.sh MobileNetV2 1 1 1 2 0 constrain
# bash script_single_model.sh MobileNetV2 1 1 1 2 0 weight[2,1,1]
```

### Multi Model Search

Here are the parameters you can set for the Multi Model Search:

- **`model_name_set`**: Specify the name of the model set you want to run.

- **`weights`**: Specify the search weights for each model in the form `[latency, power, area]`. If you have two sets of models, you can write it like `[1,1,1],[1,1,1]`. You need to make sure that the order of the weights array matches the order of the models in `model_name_set`.

- **`heterogeneity`**: This parameter defines the base number of tiles for exploration for each model. It sets the initial breadth of the search.

- **`combine_heterogeneity`**: This parameter specifies the actual number of tiles used for the combined exploration. For example, if `heterogeneity` is 2 and `combine_heterogeneity` is 3, the search explores each single model with 2 tiles and then combines these to generate combinations using 3 tiles. This creates a layered exploration strategy where individual model explorations feed into a combined model exploration.

- **`extract_num`**: Specifies the number of top-performing tiles to select from the results of the initial tile exploration. This parameter helps focus the search by continuing with only the most promising configurations.

- **`GA_TOPSIS_Weight`**: Set the weights for the TOPSIS method used during the multi-model search, formatted as `[latency, power, area]`.

- **`GA_generation`**: Specify the number of generations for the Genetic Algorithm to run.

- **`GA_population`**: Determine the number of candidates in each generation of the Genetic Algorithm. This sets the population size for the GA, influencing the diversity and convergence of the solution space.

- **`accuracy_aware`**:
  - Decide whether to include accuracy in the search criteria:
    - `0`: Exclude accuracy from the search.
    - `1`: Include accuracy in the search.

- **`guide_strategy`**:
  - Choose from two guide strategies to direct the search process:
    - **`constrain guide search`**:
      - Type `constrain` to proceed with exploring under specific constraints
      - (e.g., homogeneous architecture search results).
    - **`weight guide search`**:
      - Enter weights in the format `weight[latency, power, area]`, like `weight[2,1,1]`. This adjusts the importance of each performance metric to guide the search.

```bash
pueued -d #launch pueue to use multiple cores
bash script_multi_model.sh {model_name_set} {weights} {heterogeneity} {combine_heterogeneity} {extract_num} {GA_TOPSIS_Weight} {GA_generation} {GA_population} {accuracy_aware} {guide_strategy}

# For example
# bash script_multi_model.sh MobileNetV2,SqueezeNet [1,1,1],[1,1,1] 2 3 20 [1,1,1] 3 10 0 constrain
# bash script_multi_model.sh MobileNetV2,SqueezeNet [1,1,1],[1,1,1] 2 3 20 [1,1,1] 3 10 0 weight[2,1,1]
```

### Outputs and Logs

#### Log Directory Paths

The structure of the log directories varies depending on whether the search considers accuracy:

- **When Accuracy is Not Considered**: 
/{model_name}/accuracy_false/ADC_[#]/Cellbit_[#]/Tile_#/PE_#/SA_[#]/heterogeneity_#/TimeStamp,

e.g. MobileNetV2,SqueezeNet/accuracy_false/ADC_[5]/Cellbit_[2]/Tile_8/PE_2/SA_[128]/heterogeneity_2/2024y07M15D_16h18m

- **When Accuracy is Considered**: /{model_name}/accuracy_true/Tile_#/PE_#/SA_[#]/ADC_[#]/Cellbit_[#]/heterogeneity_#/TimeStamp

e.g. MobileNetV2,SqueezeNet/accuracy_true/Tile_8/PE_2/SA_[128]/ADC_[5]/Cellbit_[2]/heterogeneity_2/2024y07M15D_16h18m

#### Files Created

Within these directories, several files are generated to document the search parameters and results:

- **`Booksim_parameter.txt`**
- Contains the parameters used in Booksim simulations, detailing the setup for network traffic and behavior modeling.
- **`Neurosim_parameter.txt`**
- Records the parameters for Neurosim simulations.
- **`CrossSim_parameter.txt`** (only generated when accuracy is considered)
- Only accuracy-aware searches provide details about the parameters used in the CrossSim simulation.
- **`Navcim_search_result.txt`**
- Stores the final results of the search, including configurations for each model evaluated.

#### Examples of the files.

- **Booksim_parameter.txt**
```bash
#=======================================#
topology : mesh
n : 2
routing_function : dor
num_vcs : 8
vc_buf_size : 8
wait_for_tail_credit : 1
vc_allocator : islip
sw_allocator : islip
alloc_iters : 1
credit_delay : 2
routing_delay : 0
vc_alloc_delay : 1
sw_alloc_delay : 1
input_speedup : 2
output_speedup : 1
internal_speedup : 1.0
subnets : 1
traffic : neurosim
sim_type : latency
sample_period : 1
sim_power : 1
watch_out : -
#=======================================#
```
- **Neurosim_parameter.txt**
```bash
#=======================================#
operationmode : conventionalParallel (Use several multi-bit RRAM as one synapse)
memcelltype : RRAM
accesstype : CMOS_access
transistortype : conventional
deviceroadmap : LSTP
globalBufferType : register file
globalBufferCoreSizeRow : 128
globalBufferCoreSizeCol : 128
tileBufferType : register file
tileBufferCoreSizeRow : 32
tileBufferCoreSizeCol : 32
peBufferType : register file
chipActivation : activation outside Tile
reLu : reLu
novelMapping : false
SARADC : MLSA
currentMode : MLSA use CSA
pipeline : layer-by-layer process --> huge leakage energy in HP
speedUpDegree : 8
validated : validated by silicon data (wiring area in layout, gate switching activity, post-layout performance drop...)
synchronous : synchronous, clkFreq will be decided by sensing delay
algoWeightMax : 1
algoWeightMin : -1
clkFreq : 1e9
temp : 300
technode : 22
#=======================================#

```

- **CrossSim_parameter.txt**
```bash
#=======================================#
imagenet sim: 100 images, start: 0
Mapping: BALANCED
  Subtract current in crossbar: True
  Weight quantization: 8 bits
  Digital bias: True
  Batchnorm fold: True
  Bias quantization off
  Programming error off
  Read noise off
    ADC range option: CALIBRATED
    ADC topology: generic
  Activation quantization on, 8 bits
  Input bit slicing: False
  Parasitics off
  On off ratio: 1000.0
  Weight drift off
#=======================================#
```

- **Navcim_search_result.txt**
```bash
# Setup Information and Parameters
Model          : MobileNetV2, SqueezeNet
Weight         : Latency = 1, Power = 1, Area = 1
Heterogeneity  : 2
GA Generation  : 5
GA Population  : 10
Accuracy Aware : True
Constraint (user input) : Latency < [175025615.55, 92044131.69], Power < [123.86, 94.17], Area < [43306397.86, 18791001]

# Search Space
SA Width       : 64, 128, 256
SA Height      : 64, 128, 256
PE Size        : SA Width × n (2 ≤ n ≤ 16) ∀n (m % n = 0)
Tile Size      : SA Width × m (2 ≤ m ≤ 32)
ADC Precision  : 4, 5, 6
Cellbit        : 1, 2, 4

# Search Result Summary
TOP 1 Data Log
+------------+-------------------+----+-------+-----------+---------+----------+-------+
|Model       | Tile1 & Tile2     |ADC |Cellbit|Latency(ns)|Power(mW)|Area(um^2)| Acc(%)|
+============+===================+====+=======+===========+=========+==========+=======+
|MobileNetV2 |SA_w=256, SA_h=256,|    |       |  1.328e7  |  94.95  |  2.96e7  |  66   |
|            |PE=512, Tile=1024  |    |       |           |         |          |       |
+------------|                   | 6  |   4   |-----------+---------+----------+-------+
|SqueezeNet  |SA_w=64, SA_h=64,  |    |       |  7.310e6  |  64.47  |  2.96e7  |  52   |
|            |PE=128, Tile=640   |    |       |           |         |          |       |
+------------+-------------------+----+-------+-----------+---------+----------+-------+

MobileNetV2 Layer Mapping Result
Layer 1: Tile2, Layer 2: Tile2, Layer 3: Tile2, Layer 4: Tile1,...

SqueezeNet Layer Mapping Result
Layer 1: Tile2, Layer 2: Tile2, Layer 3: Tile2, Layer 4: Tile2,...

TOP 2 Data Log
...
TOP 3 Data Log
...
```

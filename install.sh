#!/bin/bash

# Install dependencies
apt-get update
apt-get install wget gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev git python3-pip aria2 flex bison curl unzip

aria2c -q -d /tmp -o cmake-3.21.0-linux-x86_64.tar.gz  https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz
tar -zxf /tmp/cmake-3.21.0-linux-x86_64.tar.gz --strip=1 -C /usr


wget -c https://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar -xvf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
cp -rl clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04/* /usr/local
rm -rf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04 clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz


curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | bash
cargo install pueue

export NAVCIM_DIR=$(pwd)


cd $NAVCIM_DIR
git submodule update --init --recursive

cd $NAVCIM_DIR/booksim2/src
make -j 

cd $NAVCIM_DIR/Inference_pytorch/NeuroSIM
make -j 


cd $NAVCIM_DIR/tvm
git submodule update --init --recursive
mkdir build
cp $NAVCIM_DIR/config.cmake build
cd build
cmake ..
make -j 

cd $NAVCIM_DIR
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda create -n neurosim python=3.6.9
conda activate neurosim
pip install -r $NAVCIM_DIR/requirments_neurosim.txt

conda create -n navcim python=3.10
conda activate navcim
pip install -r $NAVCIM_DIR/requirments_navcim.txt
cd $NAVCIM_DIR/cross-sim
pip install .
mkdir -p $NAVCIM_DIR/cross-sim/applications/dnn/data/datasets/imagenet/
cd $NAVCIM_DIR/cross-sim/applications/dnn/data/datasets/imagenet/
# 다운로드할 URL과 파일 이름
url="https://www.dropbox.com/scl/fi/nswi46sa5hti0dhxhyzuu/imagenet.zip?rlkey=2bue10y0v1nq0gwmfiqzoou2j&st=x6wd7wmn&dl=1"
output_file="imagenet.zip"

# 파일 다운로드
echo "Downloading file..."
wget -O "$output_file" "$url"

# 다운로드가 성공적으로 완료되었는지 확인
if [ $? -ne 0 ]; then
    echo "Failed to download the file."
    exit 1
fi

# 파일이 실제로 ZIP 파일인지 확인
if file "$output_file" | grep -q "Zip archive data"; then
    echo "File is a valid ZIP archive. Unzipping..."
    unzip "$output_file"

    if [ $? -ne 0 ]; then
        echo "Failed to unzip the file."
        exit 1
    else
        echo "File unzipped successfully."
    fi
else
    echo "Downloaded file is not a valid ZIP archive."
    exit 1
fi


# add tvm path to bashrc
echo "export TVM_HOME=/root/tvm" >> /root/.bashrc
echo "export PYTHONPATH=/root/tvm/python" >> /root/.bashrc
echo "export NAVCIM_DIR=$NAVCIM_DIR" >> /root/.bashrc
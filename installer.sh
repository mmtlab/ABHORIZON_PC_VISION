#!/bin/bash

cd
echo "ubuntu version:"
lsb_release -a
echo "check connection :"
ping -q -c1 google.com &>/dev/null && echo online || echo offline
#first install python 3.6
echo  "call : << pip freeze > requirements.txt  << to generate a requirements file of the python libs to export this project"
echo "install prerequisites..."
sudo add-apt-repository ppa:deadsnakes/ppa  
sudo apt update && apt upgrade -y

sudo apt-get install libssl-dev openssl -y
sudo apt-get install build-essential checkinstall -y
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev -y
#install python version
echo "installing...python 3.8.9"
sudo apt-get install python3.8 -y

echo "Ensure pip, setuptools, and wheel are up to date"

sudo apt-get install python3-pip -y
python3.6 -m pip install --upgrade pip setuptools wheel
echo "attention: selecting the correct version of python"
sudo update-alternatives --install /usr/bin/python python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python python3 /usr/bin/python3.8 2
sudo update-alternatives --list python3
read -p "wait any key to continue"
sudo update-alternatives --set python3 /usr/bin/python3.6
sudo update-alternatives --config python3


cd /usr/bin/
sudo ln -sf python3.6 python3
cd

echo "python3 version : "
python3 --version
read -p "wait any key to continue"



cd
sudo apt-get install git -y
echo "download repo"
git clone https://github.com/mmtlab/ABHORIZON_PC_VISION.git
echo "install packages"
cd ABHORIZON_PC_VISION
sudo pip3 install -r requirements.txt
read -p "wait any key to continue"
python3 --version
pip3 install absl-py==0.13.0
pip3 install attrs==21.2.0
pip3 install dataclasses==0.8
pip3 install imutils==0.5.4
pip3 install mediapipe==0.8.3
pip3 install numpy==1.19.3
pip3 install opencv-contrib-python==4.5.3.56
pip3 install opencv-python==4.5.3.56
pip3 install protobuf==3.17.3
pip3 install six==1.16.0



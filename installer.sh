cd
#first install python 3.6
echo "install prerequisites..."
sudo add-apt-repository ppa:deadsnakes/ppa  
sudo apt update && apt upgrade -y

sudo apt-get install libssl-dev openssl -y
sudo apt-get install build-essential checkinstall
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev -y
#install python version
echo "installing...python 3.6.9"
sudo apt-get install python3.6

echo "Ensure pip, setuptools, and wheel are up to date"

sudo apt-get install python3-pip -y
python3.6 -m pip install --upgrade pip setuptools wheel
sudo update-alternatives  --set python /usr/bin/python3.6

echo "python3 version : "
python3 --version
 


echo "download repo"
git clone https://github.com/bernardolanza93/ABHORIZON_PC_VISION.git
echo "install packages"
cd ABHORIZON_PC_VISION
sudo pip3 install -r requirements.txt



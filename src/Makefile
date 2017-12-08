
DATA_DIR=${HOME}/Data//NIPS2017/images
LOCAL_DATA=./NIPS_1000
OUTPUT_DIR=./Output_Attack
EPS=4


#-------------------------------------------------------------------------------
default :
	echo "please pick an explicit target"


# first run is to train model; second run generates AE
cifar10-data :
	\rm -f ./Weights/cifar10*
	PYTHONPATH=./cleverhans python ./models/cifar10.py
	PYTHONPATH=./cleverhans python ./models/cifar10.py

cifar10-study :
	CUDA_VISIBLE_DEVICES="1", PYTHONPATH=./cleverhans python ./cifar10_study.py


# Use this to create a local copy of NIPS data set
# (split into "train" and "test")
dataset :
	python split_train_test.py $(DATA_DIR) 800


# shows how to use attack codes
attack :
	\rm -rf ./Output && mkdir ./Output
	python subspace_attack.py $(LOCAL_DATA)/Test ./Output | tee ./Output/log.txt

##
## Demonstrates how to set up/execute experiment.
##
##

GPU=CUDA_VISIBLE_DEVICES="2"

DATA_DIR=${HOME}/Data//NIPS2017/images
LOCAL_DATA=./NIPS_1000
OUTPUT_DIR=./Output_Attack
EPS=4



#-------------------------------------------------------------------------------
default :
	echo "please pick an explicit target"
	echo $(GPU)


#-------------------------------------------------------------------------------
# CIFAR-10
#-------------------------------------------------------------------------------

# Trains the CIFAR-10 tensorflow example model.
# ** 
#   After training is complete, move/copy the 
#   desired checkpoint file from /tmp/cifar10_data to
#   ./Weights for use by other scripts.
# **
#
# Note: this takes a long time; you may want to do:
#     nohup make cifar10-train &
# and work on another project for awhile.
#
# We put training on GPU 1 to be polite;
# change/remove CUDA_VISIBLE_DEVICES if you don't want this
# (e.g. if you have only one GPU on your system).
# 
# Note that this CIFAR model significantly out-performs
# the sample CNN that comes with CH.
# The former should achieve ~81% accuracy on test data
# while the latter will be ~65% accuracy.
#
cifar10-train :
	$(GPU) python -m models.cifar10.cifar10_train


# Generates AE for cifar-10
cifar10-ae :
	$(GPU) python -m models.cifar10.cifar10_wrapper


# Analysis of AE generated previously/above.
# This can also take awhile, you may want to nohup this target as well.
cifar10-study :
	$(GPU) python cifar10_study.py



#-------------------------------------------------------------------------------
# NIPS-1000
#-------------------------------------------------------------------------------

# Use this to create a local copy of NIPS data set
# (split into "train" and "test")
dataset :
	python split_train_test.py $(DATA_DIR) 800


# shows how to use attack codes
attack :
	\rm -rf ./Output && mkdir ./Output
	python subspace_attack.py $(LOCAL_DATA)/Test ./Output | tee ./Output/log.txt

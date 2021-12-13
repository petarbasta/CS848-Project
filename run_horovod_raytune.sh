#!/bin/bash

read_path() {
    read -ep "Path to $1: " the_path
    echo $(readlink -f $the_path)
}

get_full_path() {
    echo $(readlink -f $1)
}

check_dir_exists() {
	if [ ! -d $1 ]; then
		echo No directory exists at $1, exiting...
		exit 1
	fi
}

check_file_exists() {
	if [ ! -f $1 ]; then
		echo No file exists at $1, exiting...
		exit 1
	fi
}


echo Welcome to Horovod+RayTune Runner!
echo
echo NOTE: You must run this script on the head node of your Ray
echo cluster, and fill in details of your machine in ray_cluster.yaml.
echo This script will start a new Ray cluster, but will not tear it
echo down. Please monitor progress and tear down manually upon
echo completion, using the following command:
echo
echo "    ray stop && ray down ray_cluster.yaml"
echo
echo =============================
echo "[1/4] Specify parameters"
echo =============================
echo

echo -n "DNN model (resnet or alexnet): "
read dnn_model

echo -n "DNN dataset / task (imagenet or mnist): "
read dataset_task

# Verify that expected files for this job exist
if [ "$dataset_task" == "imagenet" ]; then
		data_path=$(read_path "ImageNet data (CLS-LOC folder)")
		check_dir_exists $data_path
		train_path=$(get_full_path "models/ImageNet/horovod_raytune.py")
		hyp_cfg_path=$(get_full_path "models/ImageNet/hyperparameter_space_ImageNet.json")
elif [ "$dataset_task" == "mnist" ]; then
		train_path=$(get_full_path "models/MNIST/horovod_raytune.py")
		hyp_cfg_path=$(get_full_path "models/MNIST/hyperparameter_space_MNIST.json")
else
		echo Invalid DNN task $dataset_task, exiting...
		exit 1
fi

check_file_exists $train_path
check_file_exists $hyp_cfg_path

ray_cluster_yaml_path=$(get_full_path "./ray_cluster.yaml")
check_file_exists $ray_cluster_yaml_path

cur_date=$(date '+%FT%H%M%S')
log_path=./logs/run_${dataset_task}_${dnn_model}_horovod_raytune_${cur_date}.log


echo
echo =============================
echo "[2/4] Ray Tune Summary"
echo =============================
echo "DNN dataset / task:                    $dataset_task"
echo "DNN model / architecture:              $dnn_model"
echo "DNN training file:                     $train_path"
echo "DNN hyperparameter space config file:  $hyp_cfg_path"
echo "Number of DNN training epochs:         1"

if [ "$dataset_task" == "imagenet" ]; then
		echo "ImageNet data folder:                  $data_path"
fi

echo "Ray cluster config file:               $ray_cluster_yaml_path"
echo "stdout will be logged to:              $log_path"
echo

# Confirm settings with user
read -p "Please confirm these parameters by typing Y: " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
	echo The job has been cancelled, please run again. Exiting...
	exit 1
fi
echo


# Activate venv
source venv/bin/activate

echo =============================
echo "[3/4] Launch Ray Cluster"
echo =============================

echo Tearing down any current Ray cluster or nodes, please follow any prompts...
ray stop && ray down ray_cluster.yaml
echo Done!

echo Starting up a new Ray cluster, please follow the prompts...
ray up ray_cluster.yaml --no-config-cache
echo Done!

echo
echo =============================
echo "[4/4] Run DNN Training"
echo =============================
echo "Starting DNN training job..."

if [ "$dataset_task" == "imagenet" ]; then
		HOROVOD_START_TIMEOUT=600 nohup python -u $train_path \
				--data $data_path \
				--arch $dnn_model \
				--epochs 1 \
				--dnn_hyperparameter_space $hyp_cfg_path \
				--dnn_metric_key accuracy \
				--dnn_metric_objective max \
				> $log_path 2>&1 &
elif [ "$dataset_task" == "mnist" ]; then
		HOROVOD_START_TIMEOUT=600 nohup python -u $train_path \
				--arch $dnn_model \
				--epochs 1 \
				--dnn_hyperparameter_space $hyp_cfg_path \
				--dnn_metric_key accuracy \
				--dnn_metric_objective max \
				> $log_path 2>&1 &
else
		echo Invalid DNN task $dataset_task, exiting...
fi

# Deactivate venv
deactivate

echo Your training job has been started!
echo Please remember to monitor progress using the log file, and tear down your Ray cluster when finished.
echo Use this command to quickly print your log:
echo "    cat $log_path"
echo Exiting...

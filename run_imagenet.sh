#!/bin/bash

read_path() {
    read -ep "Path to $1: " the_path
    echo $(readlink -f $the_path)
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


echo Welcome to HyperTune Image Runner!
echo
echo =============================
echo "[1/4] Specify parameters"
echo =============================
echo

echo -n "Username for SSH to remote machines: "
read username

echo -n "Password for SSH to remote machines: "
read -s password
echo

echo -n "DNN model (resnet or alexnet): "
read dnn_model

echo -n "DNN parallelization strategy (dp, mp, or gpipe): "
read dnn_strategy

venv_path=$(read_path "virtual environment")
check_dir_exists $venv_path

train_path=$(read_path "DNN training file (.py)")
check_file_exists $train_path

train_cfg_path=$(read_path "DNN training file args config (.json)")
check_file_exists $train_cfg_path

hyp_cfg_path=$(read_path "DNN hyperparameter space config (.json)")
check_file_exists $hyp_cfg_path

cur_date=$(date '+%FT%H%M%S')
log_path=./logs/run_imagenet_${dnn_model}_${dnn_strategy}_${cur_date}.log


# Run controller
echo
echo =============================
echo "[2/4] Controller Summary"
echo =============================
echo "Virtual environment path:              $venv_path"
echo "DNN training file:                     $train_path"
echo "DNN training config file:              $train_cfg_path"
echo "DNN hyperparameter space config file:  $hyp_cfg_path"
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

echo "[3/4] Starting Controller..."

# Activate venv
source venv/bin/activate

# Run controller
nohup python -u controller/controller.py \
        --venv $venv_path \
        --dnn $train_path  \
        --dnn_hyperparameter_space $hyp_cfg_path \
        --dnn_train_args $train_cfg_path \
        --dnn_metric_key accuracy \
        --dnn_metric_objective max \
        --username $username \
	--password $password \
	--debug \
        --machines gpu1 gpu2 gpu3 \
        > $log_path 2>&1 &

# Deactivate venv
deactivate

echo "[4/4] Controller is now running in a separate process! Exiting..."


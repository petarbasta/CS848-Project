#!/bin/bash

# Read Username
echo -n "Username for SSH to remote machines: "
read username

# Read Password
echo -n "Password for SSH to remote machines: "
read -s password
echo

cur_date=$(date '+%FT%H%M%S')
log_path="./logs/run_imagenet_resnet_dp_$cur_date.log"
echo "stdout will be logged to $log_path"

# Run controller
echo "Starting Controller..."

# Activate venv
source venv/bin/activate

# Run controller
nohup python -u controller/controller.py \
        --venv /u4/jerorset/cs848/CS848-Project/venv \
        --dnn /u4/jerorset/cs848/CS848-Project/models/ImageNet/train.py  \
        --dnn_hyperparameter_space /u4/jerorset/cs848/CS848-Project/models/ImageNet/hyperparameter_space_resnet.json \
        --dnn_train_args /u4/jerorset/cs848/CS848-Project/models/ImageNet/train_args_dp_resnet.json \
        --dnn_metric_key accuracy \
        --dnn_metric_objective max \
        --username $username \
	--password $password \
	--debug \
        --machines gpu1 gpu2 gpu3 \
        > $log_path 2>&1 &

# Deactivate venv
deactivate

echo "Controller is now running in a separate process!"
echo "Exiting..."

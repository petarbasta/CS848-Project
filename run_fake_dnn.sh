#!/bin/bash

# Read Username
echo -n "Username for SSH to remote machines: "
read username

# Read Password
echo -n "Password for SSH to remote machines: "
read -s password
echo

cur_date=$(date '+%FT%H%M%S')
log_path="./logs/run_fake_dnn_$cur_date.log"
echo "stdout will be logged to $log_path"

# Run controller
echo "Starting Controller..."

# Activate venv
source venv/bin/activate

# Run controller
nohup python -u controller/controller.py \
        --venv /u5/p3basta/CS848/CS848-Project/venv \
        --dnn /u5/p3basta/CS848/CS848-Project/models/Fake/train.py  \
        --dnn_hyperparameter_space /u5/p3basta/CS848/CS848-Project/models/Fake/hyperparameter_space.json \
        --dnn_train_args /u5/p3basta/CS848/CS848-Project/models/Fake/fake_train_args.json \
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


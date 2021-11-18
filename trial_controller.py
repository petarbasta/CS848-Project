# Trial Controller schedules and coordinates trials across available machines

import argparse
import subprocess
import os

"""
python trial_controller.py \
        --venv_dir /u4/jerorset/cs848/CS848-Project/venv \
        --train_file /u4/jerorset/cs848/CS848-Project/tutorials/pytorch_model_parallel_existing.py \
        --remote_username jerorset \
        --remote_machines gpu1 gpu2 gpu3
"""

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote_username', required=True, type=str, help='Username for SSH to remote machines')
    parser.add_argument('--remote_machines', required=True, nargs='+', help='All remote machines to utilize')
    parser.add_argument('--venv_dir', required=True, type=str, help='The venv directory')
    parser.add_argument('--train_file', required=True, type=str, help='The Python file containing the PyTorch training job')

    args = parser.parse_args()
    return args


def main():
    args = init_args()
    
    venv_cmd = f"source {os.path.join(args.venv_dir, 'bin/activate')}"
    train_cmd = f"python {args.train_file}"
    
    # Sequentially loop through machines and execute training job on each
    for machine in args.remote_machines:
        ssh_cmd = f"ssh {args.remote_username}@{machine} -o StrictHostKeyChecking=no"
        full_cmd = f"{ssh_cmd} \"{venv_cmd} && {train_cmd}\""

        print(f"Logging into {machine} and beginning training...")
        result = subprocess.run(full_cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE)
        output = '\n'.join(result.stdout.splitlines())
        print(f"{machine}: {output}")


if __name__ == '__main__':
    main()


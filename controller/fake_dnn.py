import argparse
import random

"""
python fake_dnn.py --fake_arg_1 "Some Value" --fake_arg_2 "Another Value"
"""

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_arg_1', required=True, type=str, help='Argument (hyperparameter) #1 for the fake DNN')
    parser.add_argument('--fake_arg_2', required=True, type=str, help='Argument (hyperparameter) #2 for the fake DNN')

    args = parser.parse_args()
    return args


def main():
    args = init_args()
    fake_score = round(random.uniform(0,1), 3)
    print(f"[FAKE_DNN.PY {args.fake_arg_1} {args.fake_arg_2}] Accuracy is {fake_score}")


if __name__ == '__main__':
    main()


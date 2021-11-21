import argparse

"""
python fake_dnn.py --dnn_arg "Some Value"
"""

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_arg', required=True, type=str, help='An argument (hyperparameter) for the fake DNN')

    args = parser.parse_args()
    return args


def main():
    args = init_args()
    print(f"[FAKE_DNN.PY {args.dnn_arg}] Accuracy is x.xx")


if __name__ == '__main__':
    main()


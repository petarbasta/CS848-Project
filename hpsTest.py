from experiment import HyperparameterSpace

hyperparameterDict = {
  'A': [0.1, 0.2, 0.3 ,0.4],
  'B': [4,5,6,7],
  'C': [7,8,9]
}

hyperparameterSpace = HyperparameterSpace(hyperparameterDict)

for hyperparameterConfig in hyperparameterSpace:
    print(hyperparameterConfig._config_dict)
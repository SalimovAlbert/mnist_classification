import sys
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

from command_line_parser import parse_args_evaluate
from load_data import DataLoaderCSV
from config import config

def main(argv):
    ground_truth_path, predictions_path = parse_args_evaluate(argv)
    data_loader = DataLoaderCSV()
    ground_truth = data_loader.load_classes(ground_truth_path)
    predictions = data_loader.load_classes(predictions_path)
    print("Accuracy: ", accuracy_score(ground_truth, predictions))
    print(pd.DataFrame(confusion_matrix(ground_truth, predictions),
                       index=range(config['num_classes']), columns=range(config['num_classes'])))

if __name__ == "__main__":
    main(sys.argv[1:])

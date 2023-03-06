import sys
import numpy as np

from command_line_parser import parse_args_mnist
from load_data import DataLoaderCSV
from tf_train_evaluate import train, evaluate
from config import config

def main(argv):
    mode, dataset_path, input_path, output_path, model_path = parse_args_mnist(argv)
    data_loader = DataLoaderCSV()
    if mode == "train":
        image_data = data_loader.load_input(dataset_path)
        image_classes = data_loader.load_classes(dataset_path)
        train(image_data, image_classes, epochs=config['training']['num_epochs'],
              validation_split=config['training']['validation_split'], save_path=model_path)
    elif mode == "inference":
        image_data = data_loader.load_input(input_path)
        img_height, img_width = config['image']['height'], config['image']['width']
        digits = evaluate(image_data, save_path=model_path)
        data_loader.save_output_data(output_path, digits)

if __name__ == "__main__":
    main(sys.argv[1:])


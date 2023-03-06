# mnist_classification
Example project for Technical Recruiters

# README

## Overview

This project is a implementation of a convolutional neural network (CNN) for the task of handwritten digit recognition on the MNIST dataset. The model is implemented using TensorFlow 2 and trained on the MNIST dataset. The user can train the model and save it to a file or load a trained model from a file and use it for inference on new data from the console with the script`mnist.py`. The performance of the model can be evaluated using the provided `evaluate.py` script.

## Requirements
-   TensorFlow 2
-   Pandas
-   Sklearn
-   Numpy
-   OpenCV
Specific requirements are listed in `environment.yml`. To create the enviroment run `conda env create -f environment.yml`

## Usage

### Training

To train the model, run the command `python mnist.py --mode train --dataset /path/to/train.csv --model /path/to/save/model`

The train.csv file should contain two columns:

-   The path to the image file
-   The label of the image

The `--model` flag is used to specify the path to save the trained model.

### Inference

To use the trained model for inference on new data, run the command `python mnist.py --mode inference --model /path/to/trained/model --input /path/to/input.csv --output /path/to/predictions.csv`

The input.csv file should contain column 'Image path' containing the path to image files.

The `--model` flag is used to specify the path to the trained model. The `--input` flag is used to specify the path to the input csv file and the `--output` flag is used to specify the path to save the predictions.

### Evaluation

To evaluate the performance of the model on a test set, run the command `python evaluate.py --ground-truth /path/to/ground-truth.csv --predictions /path/to/predictions.csv`.

The ground-truth.csv file should contain column 'Class' with labels of images.
The predictions.csv file should contain columns 'Class' with predicted labels of images.

The script will output the accuracy of the model and a confusion matrix.

### Configuration

The model's architecture and training parameters can be configured in the `config.json` file.

## Note

The model was trained and tested on the MNIST dataset. It may need to be retrained and the parameters may need to be adjusted for other datasets and tasks.

## File Descriptions

### `mnist.py`

This is the main script file that uses the other modules to train and evaluate the model. It uses the `command_line_parser.py` module to parse command line arguments,
the `load_data.py` module to load data, the `tf_train_evaluate.py` module to train and evaluate the model, and the `tf_model.py` module to create the model.
It can be run in either "train" or "inference" mode.

### `evaluate.py`

This script uses the `command_line_parser.py` module to parse command line arguments, the `load_data.py` module to load the ground truth and predictions data, and the
`sklearn.metrics` library to calculate the accuracy and confusion matrix of the predictions.

### `config.py`

This file contains the configuration for the project, including the default command line arguments, the number of classes, and the image dimensions. It loads a `config.json` file that contains all the configurations.

### `command_line_parser.py`

This file contains functions that parse command line arguments. `parse_args_mnist` is used by the `mnist.py` script to parse arguments for the mode, dataset path,
input path, output path, and model path. `parse_args_evaluate` is used by the `evaluate.py` script to parse arguments for the ground truth path and predictions path.

### `config.json`

This is a json file that contains all the configuration for the project. It contains the default command line arguments, the number of classes, and the image dimensions.

### `load_data.py`

This file contains the `DataLoaderCSV` class, which is used to load data from CSV files. It has methods for loading the input images, loading the class labels,
and saving the output digits.

### `tf_train_evaluate.py`

This file contains the `train` and `evaluate` functions that are used to train and evaluate the model, respectively. The `train` function takes in the training data,
the training labels, and other training parameters, and returns a trained model. The `evaluate` function takes in the test data and a trained model, and returns the
predicted digits.

### `tf_model.py`

This file contains the `CNN` function that creates a convolutional neural network model. It takes in several parameters, including the output shape, scale,
compounding dropout, gaussian noise, dropout, weight decay, and batch momentum. It returns a `tf.keras.Model` object.

### `get_img_path_csv.py`

    Helper program to make csv file with image paths and classes. To use it, run it in the command line. First pass the path to folders with images, then filename.
    Each image should be in folder with the name same as it's class.

### `model_validate.py`

Helper file to open images and model predictions.

### `.gitignore`

Exclude sertain files from commiting to git.

### `model/model.h5`

TensorFlow model saved after training.

### `environment.yml`

File containing enviroment requrements. To create the enviroment run `conda env create -f environment.yml`

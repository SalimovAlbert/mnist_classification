import sys, getopt

from config import config

def parse_args_mnist(argv: list) -> tuple[str, str, str, str, str]:
    """Parses command line arguments and returns the values for mode, dataset_path, input_path, output_path, and model_path.

    Args:
        argv (list): a list of command line arguments

    Returns:
        Tuple[str, str, str, str, str]: a tuple containing the values for mode, dataset_path, input_path, output_path, and model_path.
    """
    PARAMETERS_EXAMPLE = "./mnist.py --mode train --dataset /path/to/train.csv --model /path/to/model" \
            " or ./mnist.py --mode inference --model /path/to/model --input /path/to/input.csv --output /path/to/predictions.csv"
    
    # Default values
    mode = 	config['default_command_args']['mode'],
    dataset_path = config['default_command_args']['dataset_path']
    input_path = config['default_command_args']['input_path']
    output_path = config['default_command_args']['output_path']
    model_path = config['default_command_args']['model_path']

    try:
        # getopt.getopt is used to parse command line options and arguments.
        opts, args = getopt.getopt(argv,"h",["mode=","dataset=","model=", "input=", "output="])
    except getopt.GetoptError:
        # If the format is wrong, this exception will be raised
        print('Wrong command format. Please use:')
        print(PARAMETERS_EXAMPLE)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            # Print the help command
            print ("\'--mode train\' for training and saving the model")
            print("\'--mode inference\' is for running the model and saving outputs to a file")
            print("Path to dataset contrains csv file with first column containing paths to images")
            print("and the second column containing the class. Same formatting goes to output.")
            print("Input is csv containing only the column with image paths.")
            print(PARAMETERS_EXAMPLE)
            sys.exit()

        elif opt == "--mode":
            # Check if the mode passed is either "train" or "inference"
            if arg in ["train", "inference"]:
                mode = arg
                continue
            else:
                print('Wrong command format. Please use:')
                print(PARAMETERS_EXAMPLE)
                sys.exit(2)
        elif opt == "--model":
            # Update the model path
            model_path = arg
            continue
        if mode == "train":
            if opt == "--dataset":
                # Update the dataset path
                dataset_path = arg
            else:
                print('Wrong command format. Please use:')
                print(PARAMETERS_EXAMPLE)
                sys.exit(2)

        elif mode == "inference":
            if opt == "--input":
                # Update the input path
                input_path = arg
            elif opt == "--output":
                # Update the output path
                output_path = arg
            else:
                print('Wrong command format. Please use:')
                print(PARAMETERS_EXAMPLE)
                sys.exit(2)
    return mode, dataset_path, input_path, output_path, model_path

def parse_args_evaluate(argv: list) -> tuple[str, str]:
    """Parses command line arguments and returns the values for ground truth path and predictions path.

    Args:
        argv (list): a list of command line arguments

    Returns:
        Tuple[str, str]: a tuple containing the values for ground truth path and predictions path.
    """
    PARAMETERS_EXAMPLE = "./evaluate.py --ground-truth /path/to/ground-truth-test.csv --predictions /path/to/predictions.csv"

    # Default values
    ground_truth_path = 	config['default_command_args']['ground_truth_path']
    predictions_path = config['default_command_args']['predictions_path']

    try:
        # getopt.getopt is used to parse command line options and arguments.
        opts, args = getopt.getopt(argv,"h",["ground-truth=","predictions="])
    except getopt.GetoptError:
        # If the format is wrong, this exception will be raised
        print('Wrong command format. Please use:')
        print(PARAMETERS_EXAMPLE)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            # Print the help command
            print ("To evaluate the model enter paths to two csv files containing")
            print("ground truth classes for the images and file with model predictions.")
            print(PARAMETERS_EXAMPLE)
            sys.exit()

        if opt == "--ground-truth":
            # Update the ground_truth path
            ground_truth_path = arg
        elif opt == "--predictions":
            # Update the predictions path
            predictions_path = arg
        else:
            print('Wrong command format. Please use:')
            print(PARAMETERS_EXAMPLE)
            sys.exit(2)
    return ground_truth_path, predictions_path
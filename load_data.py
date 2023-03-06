import cv2
import numpy as np
import pandas as pd
from typing import Optional

class DataIOInterface:
    def load_image_data(self, image_paths: np.ndarray[str]) -> np.ndarray[np.float32]:
        """Loads the image data from a list of image paths.
        
        Args:
            image_paths (np.ndarray[str]): an np array of image paths
        
        Returns:
            np.ndarray[np.float32]: an array of image data in float32 format
        """
        raise NotImplementedError
    
    def load_input(self, input_path: str) -> np.ndarray[np.float32]:
        """Loads the input data from the paths contained in input.
    
        Args:
            input_path (str): the path to the input
    
        Returns:
            np.ndarray[np.float32]: an array of input data
        """
        raise NotImplementedError

    def load_input_path(self, input_path: str) -> np.ndarray[str]:
        """Loads the image paths into a list of strings from the input.
    
        Args:
            input_path (str): the path to the input containing image paths
    
        Returns:
            np.ndarray[str]: a np array of image paths
            """
        raise NotImplementedError

    def load_classes(self, classes_path: str) -> np.ndarray[int]:
        """Loads the image classes into a np array of ints from the file by classes_path.
    
        Args:
            classes_path (str): the path to the file containing image classes
    
        Returns:
            np.ndarray[int]: a list of image classes
        """
        raise NotImplementedError

    def save_output_data(self, output_path: str, class_predictions: list[int]) -> None:
        """Saves the output data to the specified output path.
    
        Args:
            output_path (str): the path to save the output data
            class_predictions (list[int]): a list with class predictions
    
        Returns:
            None
        """
        raise NotImplementedError

class DataLoaderCSV(DataIOInterface):
    def load_image_data(self, image_paths: np.ndarray[str]) -> np.ndarray[np.float32]:
        """Loads the image data from a list of image paths.
        
        Args:
            image_paths (np.ndarray[str]): an np array of image paths
        
        Returns:
            np.ndarray[np.float32]: an array of image data in float32 format
        """
        image_data = []
        try:
            for image_path in image_paths:
                image_data.append(cv2.imread(image_path, cv2.COLOR_BGR2RGB)) 
                # image_data[-1] = cv2.resize(image_data[-1], (config['image']['height'], config['image']['width']), 
                #                   interpolation=cv2.INTER_AREA) # if images have different size
        except Exception as e:
            print("Could not read image data")
            raise e
        image_data = np.array(image_data, dtype=np.float32)
        image_data /= 255.0
        image_data = np.expand_dims(image_data, axis=-1)
        return image_data

    def load_input(self, input_path: str) -> np.ndarray[np.float32]:
        """Loads the input data from the paths contained in csv.
    
        Args:
            input_path (str): the path to the input csv
    
        Returns:
            np.ndarray[np.float32]: an array of input data
        """
        try:
            input_paths = self.load_input_path(input_path)
        except Exception as e:
            print("Could not read input data")
            raise e
        image_data = self.load_image_data(input_paths)
        return image_data

    def load_input_path(self, input_path: str) -> np.ndarray[str]:
        """Loads the image paths into a np array of strings from the csv.
    
        Args:
            input_path (str): the path to the csv containing image paths
    
        Returns:
            np.ndarray[str]: a np array of image paths
        """
        try:
            return np.array(pd.read_csv(input_path)['Image path'], dtype=str)
        except Exception as e:
            print("Could not read input path")
            raise e

    def load_classes(self, classes_path: str) -> np.ndarray[int]:
        """Loads the image classes into a np array of ints from the csv.
    
        Args:
            classes_path (str): the path to the csv containing image classes
    
        Returns:
            np.ndarray[int]: a list of image classes
        """
        try:
            return np.array(pd.read_csv(classes_path)['Class'], dtype=int)
        except Exception as e:
            print("Could not read classes path")
            raise e

    def save_output_data(self, output_path: str, class_predictions: list[int]) -> None:
        """Saves the output data to the specified output path in the csv format.
    
        Args:
            output_path (str): the path to save the output data
            class_predictions (list[int]): a list with class predictions
    
        Returns:
            None
        """
        try:
            pd.DataFrame(class_predictions, columns=['Class']).to_csv(output_path, index=False)
        except Exception as e:
            print("Could not write to output")
            raise e

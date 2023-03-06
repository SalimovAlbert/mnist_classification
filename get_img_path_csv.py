import os
import sys
import csv

def main(args):
    """
    Helper program to make csv file with image paths
    and classes.
    To use it, run it in the command line. 
    First pass the path to folders with images,
    then filename. Each image should be in folder with the
    name same as it's class.
    """
    filename = 'train.csv'
    if len(args) == 0:
        path = "."
    elif len(args) == 1:
        path = args[0]
    else:
        path = args[0]
        filename = args[1]
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image path', 'Class'])
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                img_class = os.path.split(os.path.dirname(file_path))[-1]
                writer.writerow([file_path, img_class])

if __name__ == "__main__":
    main(sys.argv[1:])

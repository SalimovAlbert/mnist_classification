import matplotlib.pyplot as plt
import numpy as np
import random

def plot_digit(images, digits, hw):
    plt.figure(figsize=(16, 10))
    for i, (image, digit) in enumerate(zip(images, digits)):
        plt.subplot(hw[0], hw[1], i + 1)
        plt.imshow(image, cmap=plt.get_cmap('gray'))
        plt.title(f"Digit: {digit}")
        plt.xticks([])
        plt.yticks([])
    plt.show()

def show_digit_predictions(image_data, digits):
    hw = (4,5) # amount of rows and columns to display
    chose = random.sample(range(len(image_data)), k=hw[0]*hw[1])
    plot_digit(image_data[chose], digits[chose], hw)
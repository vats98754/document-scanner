import numpy as np
import matplotlib.pyplot as plt
import cv2

class FromScratchSobel():
    def __init__(self):
        # Sobel x-operator (detects vertical edges)
        # Focuses on horizontal change with Gaussian weights in vertical direction
        gaussian_vertical_filter = np.array([[1], [2], [1]], dtype=np.int8)
        x_derivative = np.array([[-1, 0, 1]], dtype=np.int8)
        self.sobel_x = np.dot(gaussian_vertical_filter, x_derivative)
        
        # Sobel y-operator (detects horizontal edges)  
        # Focuses on vertical change with Gaussian weights in horizontal direction
        y_derivative = np.array([[-1], [0], [1]], dtype=np.int8)
        gaussian_horizontal_filter = np.array([[1, 2, 1]], dtype=np.int8)
        self.sobel_y = np.dot(y_derivative, gaussian_horizontal_filter)

        self.magnitude_matrix = np.array([], dtype=np.int16)
        self.direction_matrix = np.array([], dtype=np.int16)

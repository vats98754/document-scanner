import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

class FromScratchGaussianBlur():
    def __init__(self, ksize=(3,3), sigmaX=1.0):
        self.ksize = ksize
        self.sigmaX = sigmaX
        self.kernel = myGaussianKernel(self.ksize, self.sigmaX)
    
def gaussian(x, sigmaX=1, mu=0):
    return (1 / (sigmaX * (2*math.pi)**0.5) * math.e ** (-0.5 * ((x-mu)/sigmaX)**2))

def myGaussianKernel(ksize=(3,3), sigmaX=1.0):
    # get the center of the kernel
    center_y = (ksize[0]-1)/2
    center_x = (ksize[1]-1)/2
    sum = 0 # for normalization
    kernel = []
    for y in range(ksize[0]):
        row = []
        for x in range(ksize[1]):
            delta_y = y - center_y
            delta_x = x - center_x
            gaussian_x_y = gaussian((delta_x**2 + delta_y**2)**0.5, sigmaX)
            sum += gaussian_x_y
            row.append(gaussian_x_y)
        kernel.append(row)
    kernel_np = np.array(kernel, dtype=float)
    return kernel_np/sum
    
if __name__ == "__main__":
    kernel_height = 5
    kernel_width = 5
    sigmaX = 2.0
    gb = FromScratchGaussianBlur(ksize=(kernel_height, kernel_width), sigmaX=sigmaX)
    gk_height = cv2.getGaussianKernel(kernel_height, sigmaX)
    gk_width = cv2.getGaussianKernel(kernel_width, sigmaX)
    rgb = np.outer(gk_height, gk_width)
    print(gb.kernel)
    print(rgb)
    print("diff", gb.kernel-rgb)
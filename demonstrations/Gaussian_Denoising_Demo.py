import numpy as np
import cv2
import bm3d
import os
import skimage
import scipy.ndimage as ndimage
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
from denoiser import PlugPlayADMM_deblur, proj

h = cv2.getGaussianKernel(9, 1)
h = np.outer(h, h.transpose())
directory = os.fsencode('../images')
images = []
display = []
for file in os.listdir(directory):
    images.append(cv2.imread('../images/' + file.decode(),cv2.IMREAD_GRAYSCALE)/255)

image = images[0]
y = np.array(proj(skimage.util.random_noise(image, mode='gaussian', mean=0, var=0.01),[0,1]))
X = PlugPlayADMM_deblur(y, h, gamma=5, rho=1, tol=0.001, method = 'Median')
plt.figure(figsize=(8,5))
plt.subplot(1,2,1) 
plt.imshow(y, cmap='gray')
plt.subplot(1,2,2) 
plt.imshow(X[0], cmap='gray')
plt.show()
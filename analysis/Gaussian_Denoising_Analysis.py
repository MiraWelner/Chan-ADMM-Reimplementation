import numpy as np
import cv2
from PIL import Image, ImageFilter
import scipy as sp
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import bm3d
from scipy.signal import fftconvolve
import math
import os
import skimage
from tqdm import tqdm
import scipy.ndimage as ndimage
import pickle
import sys
sys.path.append('..')

from denoiser import PlugPlayADMM_deblur, proj
from plot_results import plot_figs

def psnr(x,y):
    """
    The Peak Signal Noise ratio is a metric of denoising - a high PSNR is better
    """
    return -10*math.log10( np.mean( (x[:]-y[:])**2) )


h = cv2.getGaussianKernel(9, 1)
h = np.outer(h, h.transpose())
directory = os.fsencode('../images')
images = []
display = []
for file in os.listdir(directory):
    images.append(cv2.imread('../images/' + file.decode(),cv2.IMREAD_GRAYSCALE)/255)


def get_data(method):
    """
    This gets the result of the plug and play deblur for different combinations
    of hyperparameters
    """
    combos = []
    scores = []
    scores_std = []
    itts = []
    itts_std = []
    for gamma in tqdm(np.arange(1,6,1)):
        sub_scores = []
        sub_itts = []
        for rho in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
            for var in np.arange(0.05, 0.105, 0.01):
                for image in images:
                    y = np.array(proj(skimage.util.random_noise(image, mode='gaussian', mean=0, var=var),[0,1]))
                    X = PlugPlayADMM_deblur(y, h, gamma=gamma, rho=rho, tol=0.001, method = method)
                    sub_scores.append(psnr(X[0].tolist(),image))
                    sub_itts.append(X[1])
            scores.append(np.mean(sub_scores))
            itts.append(np.mean(sub_itts))
            scores_std.append(np.std(sub_scores))
            itts_std.append(np.std(sub_itts))
            combos.append(r"$\rho=$" + str(f"{rho:.0e}") + ", " + r"$\gamma=$" + str(gamma))
    return combos,scores,scores_std,itts,itts_std

def get_baseline(method):
    """
    This gets the result of the plug and play deblur for different combinations
    of hyperparameters
    """
    scores = []
    for var in tqdm(np.arange(0.05, 0.105, 0.01)):
        for image in images:
            y = np.array(proj(skimage.util.random_noise(image, mode='gaussian', mean=0, var=var),[0,1]))
            if method == 'Non-Local Means':
                x = skimage.restoration.denoise_nl_means(np.array(y))
            elif method == 'Total Variation':
                x = skimage.restoration.denoise_tv_chambolle(np.array(y))
            elif method == 'BM3D':
                x = bm3d.bm3d(y,0.001)
            elif method == 'Median':
                x = ndimage.median_filter(y, size=3)
                
    scores.append(psnr(x.tolist(),image))
    return np.mean(scores), np.std(scores)


# The four different tested methods are called
median_baseline = get_baseline('Median')
nlm_baseline = get_baseline('Non-Local Means')
BM3D_baseline = get_baseline('BM3D')
TV_baseline = get_baseline('Total Variation')

# Uncomment this if you want to pull data from a pickle in the github pickles folder rather than generate it yourself
with open("../pickles/gaussian_denoising.pickle", "rb") as f:
    (median_data, nlm_data, BM3D_data, TV_data) = pickle.load(f)
"""


# Comment this if you want to pull data from a pickle in the github pickles folder rather than generate it yourself
median_data = get_data('Median')
nlm_data = get_data('Non-Local Means')
BM3D_data = get_data('BM3D')
TV_data = get_data('Total Variation')
"""


#save your data for later
with open("../pickles/gaussian_denoising.pickle", "wb") as f:
    pickle.dump((median_data, nlm_data, BM3D_data, TV_data), f)

    
#Plot your answers
#plot your data!
plot_figs(median_data, median_baseline, nlm_data, nlm_baseline, BM3D_data, BM3D_baseline, TV_data, TV_baseline, 'gaussian_denoising', "Gaussian")
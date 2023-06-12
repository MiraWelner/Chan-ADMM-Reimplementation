import numpy as np
from matplotlib import pyplot as plt
import skimage
import sys
sys.path.append('..')
from spectra_handler import *
from denoiser import Plug_And_Play_ADMM_1D


def mse(x,y):
    return (np.array(x)-np.array(y))**2
directory = "../spectra"
X = Multi_Spectra(directory=directory, n=10).matrix

#Get spectra
original_spectrum = pre.MinMaxScaler().fit_transform(np.array(X.iloc[:,4]).reshape(-1, 1))
noisy_spectrum = original_spectrum+np.random.normal(0,0.01,len(original_spectrum)).reshape(-1, 1)
tv_denoise = pre.MinMaxScaler().fit_transform(skimage.restoration.denoise_tv_chambolle(np.array(noisy_spectrum)))
solution = Plug_And_Play_ADMM_1D(noisy_spectrum, 2.1, 0.1, 0.00001, method = 'Median')[0];

#Plot results
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs[0, 0].plot(original_spectrum, 'tab:orange')
axs[0, 0].set_title('Original Dimethyl Ester FTIR Spectra')
axs[0, 1].plot(noisy_spectrum, 'tab:orange')
axs[0, 1].set_title('Dimethyl Ester with Gaussian Noise of Variance 0.01')
axs[1, 0].plot(tv_denoise, 'tab:orange')
axs[1, 0].set_title('Spectra Denoised with TV filter of weight 0.05')
axs[1, 1].plot(solution, 'tab:orange')
axs[1, 1].set_title('Spectra Denoised with plug and Play ADMM ' + r"$ \gamma=1.1$" + ", " + r"$\rho_0=0.1$")
plt.show()
import numpy as np
import os
import skimage
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from sklearn import preprocessing as pre
import pickle
from sklearn.metrics import mean_squared_error
import random
import sys
sys.path.append('..')

from denoiser import Plug_And_Play_ADMM_1D, proj
from spectra_handler import Spectrum, Multi_Spectra

directory = "../spectra"
n = 100
X = Multi_Spectra(directory=directory, n=n).matrix
spectra = [pre.MinMaxScaler().fit_transform(np.array(X.iloc[:,i]).reshape(-1, 1)) for i in range(n)]
noisy_spectra = [spectra[i]+np.random.normal(0,0.01,880).reshape(-1, 1) for i in range(n)]

def get_data(method):
    """
    This gets the result of the plug and play deblur for different combinations
    of hyperparameters
    """
    tol = 0.01
    scores_avg = []
    scores_std = []
    itts_avg = []
    itts_std = []
    combos = []
    for gamma in tqdm(np.arange(1.05,1.1,0.01)):
        for rho in np.arange(1.3, 1.4, 0.02):
            scores = []
            itts = []
            for itr, spectrum in enumerate(spectra):
                y = noisy_spectra[itr]
                solution = Plug_And_Play_ADMM_1D(y, gamma, rho, tol, method = method)
                score = mean_squared_error(solution[0], spectrum)
                scores.append(score)
                itts.append(solution[1])
            combos.append(r"$\rho=$" + str(round(rho,2)) + ", " + r"$\gamma=$" + str(gamma))
            scores_avg.append(np.mean(scores))
            scores_std.append(np.std(scores))
            itts_avg.append(np.mean(itts))
            itts_std.append(np.std(itts))
    return combos, scores_avg, scores_std, itts_avg, itts_std


def get_baseline(method):
    """
    This gets the result of the plug and play deblur for different combinations
    of hyperparameters
    """
    scores = []
    for itr, spectrum in enumerate(spectra):
        y = noisy_spectra[itr]
        if method == 'Non-Local Means':
            x = skimage.restoration.denoise_nl_means(np.array(y))
        elif method == 'Total Variation':
            x = skimage.restoration.denoise_tv_chambolle(np.array(y))
        elif method == 'Median':
            x = ndimage.median_filter(np.array(y), size=10)
                
                
        cleaned_spec = x
        if method != 'Non-Local Means':
            cleaned_spec = [x[0] for x in cleaned_spec]
        scores.append(mean_squared_error(cleaned_spec,spectrum))
    return np.mean(scores), np.std(scores)


nlm_baseline = get_baseline('Non-Local Means')
TV_baseline = get_baseline('Total Variation')
median_baseline = get_baseline('Median')


"""
# Comment this if you want to pull data from a pickle in the github pickles folder rather than generate it yourself
nlm_data = get_data('Non-Local Means')
TV_data = get_data('Total Variation')
median_data = get_data('Median')
"""

# Uncomment this if you want to pull data from a pickle in the github pickles folder rather than generate it yourself
with open('../pickles/spectra_denoising.pickle', 'rb') as handle:
    median_data, nlm_data, TV_data = pickle.load(handle)


with open("../pickles/spectra_denoising.pickle", "wb") as f:
    pickle.dump((median_data, nlm_data, TV_data), f)


#plot results
fig, ax1 = plt.subplots(figsize=(3, 7))
ax1.errorbar(median_data[1], 
             range(len(median_data[1])), 
             xerr=median_data[2], 
             label="ADMM with MF Denoiser",
            color='red')
ax1.errorbar([median_baseline[0]]*len(median_data[1]),
             range(len(median_data[1])),
             xerr=median_baseline[1], 
             label="MF Denoiser Baseline",
             linestyle = 'dashed',
             color='red')

ax1.errorbar(TV_data[1], 
             range(len(TV_data[1])), 
             xerr=TV_data[2], 
             color='orange',
             label="ADMM with TV Denoiser")

ax1.errorbar([TV_baseline[0]]*len(TV_data[1]),
             range(len(TV_data[1])),
             xerr=TV_baseline[1], 
             label="TV Denoiser Baseline",
             linestyle = '-.',
             color='orange')

ax1.errorbar(nlm_data[1], 
             range(len(nlm_data[1])), 
             xerr=nlm_data[2], 
             color='blue',
             label="ADMM with NLM Denoiser")
ax1.errorbar([nlm_baseline[0]]*len(nlm_data[1]),
             range(len(median_data[1])), 
             xerr=nlm_baseline[1], 
             color='blue',
             linestyle = '-.',
             label="NLM Denoiser Baseline")
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

plt.xlim(0,0.0007)

plt.yticks(range(len(median_data[0])), median_data[0], size='small')
plt.title("Performance Comparison of Plug-and-Play ADMM with Fixed Point Convergence Poisson Denoising Methods")
plt.xlabel("Mean Squared Error")


ax2 = fig.add_axes([1.3,0.11, 0.7, 0.75])

ax2.errorbar([i for i in median_data[3]], 
             range(len(median_data[3])), 
             xerr=[i for i in median_data[3]], 
             label="ADMM with MF Denoiser",
            color='red')
ax2.errorbar([i for i in nlm_data[3]], 
             range(len(nlm_data[3])), 
             xerr=[i for i in nlm_data[4]], 
             color='blue',
             label="ADMM with NLM Denoiser")

ax2.errorbar([i for i in TV_data[3]], 
             range(len(TV_data[3])), 
             xerr=[i for i in TV_data[4]], 
             color='orange',
             label="ADMM with TV Denoiser")
plt.yticks(range(len(median_data[0])), median_data[0], size='small')
plt.xlabel("Itterations until Convergence \n or Cutoff")

ax1.legend(bbox_to_anchor=(-1.6, 1), loc="upper left")
plt.savefig("spectral_denoising.png", bbox_inches='tight')
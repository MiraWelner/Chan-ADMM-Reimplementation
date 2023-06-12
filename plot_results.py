from matplotlib import pyplot as plt

def plot_figs(median_data, median_baseline, nlm_data, nlm_baseline, BM3D_data, BM3D_baseline, TV_data, TV_baseline, name, method):
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
                 linestyle = 'dotted',
                 color='red')

    ax1.errorbar(nlm_data[1], 
                 range(len(nlm_data[1])), 
                 xerr=nlm_data[2], 
                 color='blue',
                 label="ADMM with NLM Denoiser")
    ax1.errorbar([nlm_baseline[0]]*len(nlm_data[1]),
                 range(len(median_data[1])), 
                 xerr=nlm_baseline[1], 
                 color='blue',
                 linestyle = 'dotted',
                 label="NLM Denoiser Baseline")


    ax1.errorbar(BM3D_data[1], 
                 range(len(BM3D_data[1])), 
                 xerr=BM3D_data[2], 
                 color='green',
                 label="ADMM with BM3D Denoiser")
    ax1.errorbar([BM3D_baseline[0]]*len(BM3D_data[1]),
                 range(len(BM3D_data[1])), 
                 xerr= BM3D_baseline[1], 
                 color='green',
                 linestyle = 'dotted',
                 label="BM3D Denoiser Baseline")

    ax1.errorbar(TV_data[1], 
                 range(len(TV_data[1])), 
                 xerr=TV_data[2], 
                 color='orange',
                 label="ADMM with TV Denoiser")

    ax1.errorbar([TV_baseline[0]]*len(TV_data[1]),
                 range(len(TV_data[1])), 
                 xerr=TV_baseline[1], 
                 color='orange',
                 linestyle = 'dotted',
                 label="TV Denoiser Baseline")

    plt.yticks(range(len(median_data[0])), median_data[0], size='small')
    plt.title("Performance Comparison of Plug-and-Play ADMM with Fixed Point Convergence " + method + " Methods")
    plt.xlabel("Peak Signal-to-Noise Ratio")


    ax2 = fig.add_axes([1.3,0.11, 0.7, 0.75])

    ax2.errorbar(median_data[3], 
                 range(len(median_data[3])), 
                 xerr=median_data[4], 
                 label="ADMM with MF Denoiser",
                color='red')
    ax2.errorbar(nlm_data[3], 
                 range(len(nlm_data[3])), 
                 xerr=nlm_data[4], 
                 color='blue',
                 label="ADMM with NLM Denoiser")
    ax2.errorbar(BM3D_data[3], 
                 range(len(BM3D_data[3])), 
                 xerr=BM3D_data[4], 
                 color='green',
                 label="ADMM with BM3D Denoiser")
    ax2.errorbar(TV_data[3], 
                 range(len(TV_data[3])), 
                 xerr=TV_data[4], 
                 color='orange',
                 label="ADMM with TV Denoiser")

    plt.xlabel("Itterations until Convergence \n or Cutoff")

    ax1.legend(bbox_to_anchor=(-1.6, 1), loc="upper left")
    plt.savefig(name+".png", bbox_inches='tight')
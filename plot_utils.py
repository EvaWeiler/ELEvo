import matplotlib.pyplot as plt 
import numpy as np 

def plot_spcs_cmepropagation(spcs,cme):


    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'},layout='constrained')
    for spc in spcs:
        ax.plot(spc.longitude,spc.radial_distance,label=spc.name)
    ax.plot(cme.longitude *np.ones(cme.cme_r_ensemble.shape).flatten(),cme.cme_r_ensemble.flatten(),label="cme")
    plt.legend()
    plt.show()

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from tqdm import tqdm
import xarray as xr
import os
import scipy as sp

top_dir = '../../../data/'
C_orig = sp.sparse.load_npz(top_dir + 'connectome_sgn_cnt.npz')
C_cnt = np.abs(C_orig)
n = C_orig.shape[0]

def id_perc_steps(n_neurons, n_targets, n_sources, perturbation_rank, cnt):
    n_exp_per_step = (n_sources*n_neurons)/(perturbation_rank * n_targets)
    step_size = n_sources
    tot_count = cnt.sum()
    y = [0,] + list(np.cumsum(cnt)[::step_size]/tot_count)
    x = [0,] + list(np.arange(n_exp_per_step,   
                                n_exp_per_step*(n_neurons/n_sources + 1), 
                                n_exp_per_step))
    x,y = np.array(x), np.array(y)

    return x, y
s = 0.8
plt.figure(figsize=(4.5*s, 3*s), dpi=400)
N_out = np.array(C_cnt.sum(0)).squeeze()
#single neuron source, whole brain target

laser_dim = {'1':1, '2':2, 'source':3}
n_percs = [int(n*0.01),int(n*0.05), int(n*0.1)]
perc_widths = {int(n*0.01):1,int(n*0.05):2, int(n*0.1):4}

x, y = id_perc_steps(n, n_targets=n, n_sources=1, perturbation_rank=1, cnt=N_out)
plt.step(x/n, y, color='red', lw=4, label='1')

color = 'green'
for n_perc in n_percs:
    n_s = n_t = n_l = n_perc
    x, y = id_perc_steps(n, n_targets=n_t, n_sources=n_s, perturbation_rank=n_l, cnt=N_out)
    plt.step(x/n, y, label=n_perc, color=color, lw=perc_widths[n_perc])


plt.semilogx()
#plt.xlim(-0.1,1.1)
#change ytick labels to be percentages
plt.yticks([0,0.25,0.5,0.75,1])
plt.gca().set_yticklabels(['0', '25', '50', '75', '100'])
plt.xticks([0.0001, 0.001, 0.01, 0.1, 1])
plt.gca().set_xticklabels(['0.0001', '0.001', '0.01', '0.1', '1'], rotation=-45)
#plt.legend()
plt.grid()
plt.xlabel('# experiments/# neurons')
plt.ylabel('Effectome\nidentifiability (%)')
plt.legend(loc=(1.1,0))
plt.tight_layout()
plt.savefig('fig3_id_perc.pdf')

# %%

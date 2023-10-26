#%% import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
import numpy as np
import scipy as sp
import matplotlib as mpl
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
#%% load up connectome data
top_dir = '../../../data/'
C_orig = sp.sparse.load_npz(top_dir + 'connectome_sgn_cnt.npz')
df_meta_W = pd.read_csv(top_dir + 'meta_data.csv')
eigenvalues = np.load(top_dir + 'eigenvalues_1000.npy')
eig_vec = np.load(top_dir + 'eigvec_1000.npy')
conv_rev = pd.read_csv(top_dir + 'C_index_to_rootid.csv')
conv_dict_rev = dict(zip(conv_rev.iloc[:,0].values, conv_rev.iloc[:,1].values,))
#%% eigenvalue plot
scale = 1/np.abs(eigenvalues[0])
s = 1.1
plt.figure(figsize=(5*s,2*s), dpi=400)
plt.subplot(121)
eig_mag = np.abs(eigenvalues)*scale
sort_ind = np.argsort(eig_mag)[::-1]
eig_mag = eig_mag[sort_ind]
eig_ind = range(1, len(eigenvalues)+1)
plt.plot(eig_ind, eig_mag, label='Effectome prior', c='k', rasterized=True)
plt.loglog()
plt.xlim(.5,2e3)
plt.ylim(.5*scale,3e3*scale)
plt.xticks([1, 1e1, 1e2, 1e3])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.xlabel('Eigenvalue rank')
plt.ylabel(r'$|\lambda_i|$')

log_eig_ind = np.log(eig_ind)
X = np.vstack([log_eig_ind, np.ones(len(eig_ind))]).T
b = np.linalg.lstsq(X, np.log(eig_mag), rcond=None)[0]
plt.plot(eig_ind, np.exp(b[0]*log_eig_ind + b[1]), 'red', ls='--', 
                    label='Fit power-law ' +  r'$(\alpha=$' + str(b[0].round(2)) + ')', 
                    alpha=0.7, lw=2)
plt.legend(fontsize=7, framealpha=1)

plt.subplot(122)
#make inside of each point black and the edge white
plt.scatter(np.real(eigenvalues[sort_ind]*scale)[::-1], np.imag(eigenvalues[sort_ind][::-1])*scale, label='imag', marker='.', s=16, c='k', rasterized=True,
            edgecolors='w', lw=0.1)

lim = np.max(eig_mag)*1.1
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
#put a grid only through origin
plt.axvline(0, c='grey', alpha=0.5)
plt.axhline(0, c='grey', alpha=0.5)
yticks = xticks = np.array([-1, -0.5, 0, 0.5, 1])
plt.xticks(xticks, xticks)
plt.yticks(yticks, yticks)
#set aspect ratio to 1
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Re' r'$(\lambda_i$)')
plt.ylabel('Im' r'$(\lambda_i$)')
#plt.grid()
plt.tight_layout()
plt.savefig('./eigenvalues_1000.pdf', transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)

#%% fraction of neurons to 75% of variance plot
inds_cum = []
j=0
js = []
while j<len(eigenvalues):
    eig = eigenvalues[j]
    ev = eig_vec[:, j]
    js.append(j)
    j+=1
    if np.imag(eig)>0:
        j+=1
    
    inds = np.argsort(np.abs(ev))[::-1]
    ev = ev[inds]
    ev = np.abs(ev)**2
    ev = ev/np.sum(ev)
    ev_cum = np.cumsum(ev)
    ind = np.argmax(ev_cum>0.75)
    inds_cum.append(ind)

s = 0.9
plt.figure(figsize=(2*s, 2*s), dpi=400)
plt.scatter(range(1, len(inds_cum)+1), np.array(inds_cum), c='k', s=16, edgecolors='w', lw=0.1, rasterized=True)
plt.loglog()
plt.xlabel('Eigenvalue rank')
plt.ylabel('Number of neurons\nto 75% of magnitude')
plt.ylim(1,1e4)
#make the ylabels 1, 10, 100, 1000, etc
yticks = [1, 10, 100, 1000, 10000]
#ytick labels with commas
ytick_labels = [f'{x:,}' for x in yticks]
plt.yticks(yticks, ytick_labels)
plt.xticks(yticks[:-1], ytick_labels[:-1])
plt.grid()
plt.savefig('./eigenvalues_1000_cumsum.pdf', transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)

#%% example eigenvectors
fig, axs = plt.subplots(2,1, figsize=(3,2), dpi=400)
ev = eig_vec[:, 0]
n_ev = len(ev)
ind = np.arange(n_ev)
lim = 1.2
sub_samp = 1
neuron_index = [0, n_ev//4, n_ev//2, 3*n_ev//4]
for j, i in enumerate([0, 49]):
    max_lim = np.max(np.abs(eig_vec[:,i]))
    axs[j].scatter(ind[::sub_samp], -np.real(eig_vec[::sub_samp,i])/max_lim, label='Real', marker='.', s=5, c='k', alpha=1, zorder=100, rasterized=True)
    axs[j].scatter(ind[::sub_samp], -np.imag(eig_vec[::sub_samp,i])/max_lim, label='Imag.', marker='.', s=5, c='grey', alpha=1, rasterized=True)
    axs[j].set_ylim(-lim, lim)
    axs[j].set_xlim(0, n_ev)
    axs[j].set_xticks([1, 60000, 120000], ['1', '60,000', '120,000'])

    if j == 0:
        #tight legend
        #keep legend off plot
        axs[j].set_xticklabels([])
        axs[j].set_yticklabels([])
    elif j==1:
        axs[j].legend(loc=(1.01, 0), fontsize=6, labelspacing=0.1, borderpad=0.2, handletextpad=0.1)   

        axs[j].set_xlabel('Neuron index')
        axs[j].set_ylabel('Eigenvector loading')
    else:
        axs[j].set_xticklabels([])
        axs[j].set_yticklabels([])
    #annotate each in upper left above plot with eigenvector index
    #axs[j].annotate('Eigenvector '+str(i+1), xy=(0.01, 1.1), xycoords='axes fraction', fontsize=6, ha='left', va='top')
    #just do it as a title
    axs[j].set_title('Eigenvector '+str(i+1), fontsize=7)
plt.tight_layout()
plt.savefig('eigenvector_examples.pdf', transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)
# %% correlation between eigenvectors
eig_vecs_real = []
j=0
while j < len(eig_vec[0]):
    ev = eig_vec[:, j]
    j+=1
    if np.sum(np.imag(ev)**2)>0:
        j+=1
        eig_vecs_real.append(np.imag(ev))
        eig_vecs_real.append(np.real(ev))
    else:
        eig_vecs_real.append(np.real(ev))

eig_vecs_real = np.array(eig_vecs_real)

sort_ind = np.arange(len(eig_vecs_real))
eig_vecs_real = eig_vecs_real/np.sum((eig_vecs_real)**2, 1, keepdims=True)**0.5
eig_cov = eig_vecs_real[sort_ind] @ eig_vecs_real[sort_ind].T
plt.figure(figsize=(3,3), dpi=300)
plt.imshow(np.abs((eig_cov[:50,:50])), cmap='gray');
plt.colorbar(label='|r|', fraction=0.046, pad=0.04, orientation='vertical', ticks=[0,1], )
plt.xticks([0,49])
plt.yticks([0,49])
ax = plt.gca()
ax.set_xticklabels([1,50])
ax.set_yticklabels([1,50])
plt.xlabel('Eigenvalue rank')
plt.ylabel('Eigenvalue rank')
plt.tight_layout()
plt.savefig('./eigenvector_correlation.pdf', transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)

#%% summary of eigenvector correlations
s = 0.8
plt.figure(figsize=(3.5*s,3*s), dpi=300)
eig_cov_copy = eig_cov.copy()
eig_cov_copy[np.diag_indices_from(eig_cov)] = 0
plt.scatter(range(1,1001),np.quantile(np.abs(eig_cov_copy),0.5, 1), 
                        c='k',  label='median',s=1, rasterized=True)
plt.scatter(range(1,1001),np.quantile(np.abs(eig_cov_copy),0.99, 1),
    c=[0.3,0.3,0.3], label='99th percentile',s=1, rasterized=True)
plt.scatter(range(1,1001),np.max(np.abs(eig_cov_copy), 1),c=[0.7,0.7,0.7], 
                label='max',s=1, rasterized=True)
plt.ylim(-0.01,1)
plt.semilogx()
plt.xticks([1, 10, 100, 1000])
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.legend( loc='upper left', markerscale=3, framealpha=1, labelspacing=0.1, borderpad=0.2, handletextpad=0.1)
plt.xlabel('Eigenvalue rank')
plt.ylabel('|r|')
plt.tight_layout()
plt.savefig('mode_corr_dist.pdf')

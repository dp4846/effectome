#%%
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import scipy as sp
import matplotlib.colors as mcolors
import matplotlib as mpl
def rectify(x):
    temp_x = np.copy(x)
    temp_x[x<0] = 0
    return temp_x
#set the matplotlib font to arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
colors = ['#95A3CE', '#D5A848', '#7dcd13', '#47c4cb', '#47c4cb', '#E760A3', '#BA00FF', '#8B4513', '#39FF14', '#FF4500', '#708090']
label_colors = [mcolors.to_rgb(color) for color in colors]
#%% load up dynamics matrix, meta data, eigenvalues, and eigenvectors
top_dir = '../../../data/'
C_orig = sp.sparse.load_npz(top_dir + 'connectome_sgn_cnt.npz')
df_meta_W = pd.read_csv(top_dir + 'meta_data.csv')
eigenvalues = np.load(top_dir + 'eigenvalues_1000.npy')
eig_vec = np.load(top_dir + 'eigvec_1000.npy')
scale_orig = 1/np.abs(eigenvalues[0])
nm = top_dir + 'C_index_to_rootid.csv'
conv_rev = pd.read_csv(nm)
conv_dict_rev = dict(zip(conv_rev.iloc[:,0].values, conv_rev.iloc[:,1].values,))
# %% optic lobe experiment
dt = 10
sample_rate = 100
T = 100
ev_ind = 0
ts =  np.arange(0, T, 1/sample_rate)
all_sorted_inds = np.argsort(np.abs(eig_vec[:, ev_ind]))[::-1]
#get up to 75% power 
ev_abs = np.abs(eig_vec[all_sorted_inds, ev_ind])
frac_var_ind = np.where(np.cumsum(ev_abs**2)/np.sum(ev_abs**2)>0.75)[0][0]

top_ind = np.array([104059, 80911, 117433, 101574] + list(all_sorted_inds[4:frac_var_ind]))
label = ['VCH', 'DCH', 'LPi21', 'Am1']
x0 = -np.real(eig_vec[top_ind, 0])
W = C_orig[top_ind, :][:, top_ind].todense()*scale_orig
#%%
ts =  np.arange(0, T, 1/sample_rate)
A = sp.linalg.logm(W)/dt # log of matrix
A_step = sp.linalg.expm(A * (1/sample_rate))
xs = [x0,]#for recording linear responses
xs_rect = [x0,]#for recording rectified responses
for t in ts[1:]:
    #at each step plug the last time step in and take a 1/sample_rate sized step
    xs.append(A_step @ xs[-1])
    xs_rect.append(A_step @ rectify(xs_rect[-1]))#rectify inputs
xs = np.array(np.real(xs))
xs_rect = np.array(np.real(xs_rect))

#%%
s=0.7
N_w = 4
plt.figure(figsize=(4*s, 4*s), dpi=400)
A_disp = np.array(W[:N_w, :N_w])/scale_orig
lim = np.max(np.abs(A_disp))
plt.imshow(A_disp, cmap='RdBu_r', vmin=-lim, vmax=lim)
#xticks are label
plt.xticks(np.arange(N_w), label[:N_w], rotation=-45, fontsize=8)
plt.yticks(np.arange(N_w), label[:N_w], rotation=0, fontsize=8)
plt.colorbar(label='Synaptic count X sign')
plt.xlabel('Pre-synaptic neuron')
plt.ylabel('Post-synaptic neuron')
plt.savefig('opp_motion_weights.pdf', bbox_inches='tight')
#%%
s = 0.8
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8*s, 2.5*s), dpi=400)
ylim = 0.6
ax1.set_ylim([-ylim, ylim])
ax1.set_xlabel('Neuron index')
ax1.set_ylabel('Eigenvector loading')
#scatters are open circles
ax1.scatter(range(len(eig_vec[:,ev_ind])), -np.real(eig_vec[:,ev_ind]), color='gray', alpha=1, label='Real', marker='o', facecolors='none',s=8)
ax1.legend(loc='lower left', fontsize=8,  handletextpad=0.1, handlelength=0.5)
#annotate with eigenvalue
ax1.annotate(r'$\lambda_2$=' f'{((eigenvalues[ev_ind])*scale_orig*0.99):.2f}', xy=(0.1, 0.75), xycoords='axes fraction', fontsize=8)
for i in range(N_w):
    ax1.scatter([top_ind[i]], [-np.real(eig_vec[top_ind[i], ev_ind])], color=label_colors[i], alpha=1, marker='o', s=1)
xs = np.array(xs)
xs_rect = np.array(xs_rect)
for i in range(N_w):
    ax2.plot(ts, xs[:,i], color=label_colors[i], label=label[i], alpha=0.5, ls='--')
for i in range(N_w):
    ax2.plot(ts, xs_rect[:,i], color=label_colors[i], label=label[i], alpha=1.0, ls='-')
ax2.legend(loc='upper right', bbox_to_anchor=(1.8, 1), fontsize=7, ncol=2, handletextpad=0.1, 
                        columnspacing=0.5, title='Linear  Rectified', title_fontsize=7)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Activity (a.u.)')
ax2.set_xticks(np.arange(0, 50 + dt, dt))
ax2.set_xlim([-1, 25])
ax2.set_yticklabels([])
ax2.set_ylim([-ylim, ylim])
fig.tight_layout()
plt.savefig('opp_motion_eigenvector.pdf', bbox_inches='tight')
#%%
u_flow = np.zeros_like(x0)
u_right = np.zeros_like(x0)
x0 = np.real(eig_vec[top_ind, 0])
x0 = np.zeros_like(x0)
s = 0.001#stim strength
u_flow[np.array([0,1,2,3])] = s
u_right[np.array([0,1,])] = s
xs_rects = []
for u in ([u_right, u_flow]):
    xs_rect = [x0,]
    xs = [x0,]
    for t in ts[1:]:
        if t>30:#stop stim after 30 ms
            u[...] = 0
        xs_rect.append(A_step @ rectify(xs_rect[-1]+u))
    xs_rects.append(np.real(xs_rect))
#%%
s = 0.8
plt.figure(figsize=(3*s,2*s))
for j, xs_rect in enumerate(xs_rects):
    xs_rect = np.array(xs_rect)
    for i in range(4):
        plt.plot(ts, xs_rect[:,i], ls=['-', ':'][j], 
                        color=label_colors[i], label=[label, [' ',]*4][j][i], alpha=1)
plt.legend(ncol=2, title='BTF               BTF \nright           left + right', loc=(1.1,0.2))
plt.xlabel('Time (ms)')
plt.ylabel('Activity (a.u.)')
plt.xticks([0, 10, 20, 30, 40, 50])
plt.xlim([0, 50])
plt.yticks([0,0.5])
#plt.ylim(-0.1, 0.6)
plt.savefig('optic_lobe_stim_sim.pdf', bbox_inches='tight')

# %% ring neuron circuit
ev_ind = 41
all_sorted_inds = np.argsort(np.abs(eig_vec[:, ev_ind]))[::-1]
ev_abs = np.abs(eig_vec[all_sorted_inds, ev_ind])
frac_var_ind = np.where(np.cumsum(ev_abs**2)/np.sum(ev_abs**2)>0.75)[0][0]
top_ind = list(all_sorted_inds[:frac_var_ind])
label = list(df_meta_W.loc[top_ind]['hemibrain_type'].values)

x0 = np.real(eig_vec[top_ind, ev_ind])
W = C_orig[top_ind, :][:, top_ind].todense()*scale_orig
W  = W/(np.max(np.abs(np.linalg.eigvals(W)))*0.99)
#%%
ts =  np.arange(0, T, 1/sample_rate)
A = sp.linalg.logm(W)/dt # log of matrix
A_step = sp.linalg.expm(A * (1/sample_rate))
xs = [x0,]#for recording linear responses
xs_rect = [x0,]#for recording rectified responses
for t in ts[1:]:
    #at each step plug the last time step in and take a 1/sample_rate sized step
    xs.append(A_step @ xs[-1])
    xs_rect.append(A_step @ rectify(xs_rect[-1]))#rectify inputs
xs = np.array(np.real(xs))
xs_rect = np.array(np.real(xs_rect))

#%%
s=0.7
N_w = 10
plt.figure(figsize=(4*s, 4*s), dpi=400)
A_disp = np.array(W[:N_w, :N_w])/scale_orig
lim = np.max(np.abs(A_disp))
plt.imshow(A_disp, cmap='RdBu_r', vmin=-lim, vmax=lim)
#xticks are label
plt.xticks(np.arange(N_w), label[:N_w], rotation=-45, fontsize=8)
plt.yticks(np.arange(N_w), label[:N_w], rotation=0, fontsize=8)
plt.colorbar(label='Synaptic count X sign')
plt.xlabel('Pre-synaptic neuron')
plt.ylabel('Post-synaptic neuron')
plt.savefig('ring_circuit_weight_matrix.pdf', bbox_inches='tight')
#%%
s = 0.8
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8*s, 2.5*s), dpi=400)
ylim = 0.6
ax1.set_ylim([-ylim, ylim])
ax1.set_xlabel('Neuron index')
ax1.set_ylabel('Eigenvector loading')
#scatters are open circles
ax1.scatter(range(len(eig_vec[:,ev_ind])), np.real(eig_vec[:,ev_ind]), color='gray', alpha=1, label='Real', marker='o', facecolors='none',s=8)
ax1.legend(loc='lower left', fontsize=8,  handletextpad=0.1, handlelength=0.5)
#annotate with eigenvalue
ax1.annotate(r'$\lambda_2$=' f'{((eigenvalues[ev_ind])*scale_orig*0.99):.2f}', xy=(0.1, 0.75), xycoords='axes fraction', fontsize=8)
for i in range(N_w):
    ax1.scatter([top_ind[i]], [np.real(eig_vec[top_ind[i], ev_ind])], color=label_colors[i], alpha=1, marker='o', s=1)
xs = np.array(xs)
xs_rect = np.array(xs_rect)
for i in range(N_w):
    ax2.plot(ts, xs[:,i], color=label_colors[i], label=label[i], alpha=0.5, ls='--')
for i in range(N_w):
    ax2.plot(ts, xs_rect[:,i], color=label_colors[i], label=label[i], alpha=1.0, ls='-')
ax2.legend(loc='upper right', bbox_to_anchor=(1.8, 1), fontsize=7, ncol=2, handletextpad=0.1, 
                        columnspacing=0.5, title='Linear  Rectified', title_fontsize=7)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Activity (a.u.)')
ax2.set_xticks(np.arange(0, 50 + dt, dt))
ax2.set_xlim([-1, 25])
ax2.set_yticklabels([])
ax2.set_ylim([-ylim, ylim])
fig.tight_layout()
plt.savefig('ring_neuron_eigenvector.pdf', bbox_inches='tight')
#%%
angle = np.linspace(0,0.7,10)
c = plt.cm.hsv(angle)
T = 100
W_run  = W
W_run = 0.9*W_run/(np.max(np.abs(np.linalg.eigvals(W_run))))
ts =  np.arange(0, T, 1/sample_rate)
A = sp.linalg.logm(W_run)/dt # log of matrix
A_step = sp.linalg.expm(A * (1/sample_rate))
for stim_neur in [0,5]:
    u = np.zeros(frac_var_ind)
    s = 0.0015
    g = 0.6
    u[...] = 0  
    u[:25] = s*g
    u[stim_neur] = s
    xs_rect = [u,]
    xs = [u,]
    for t in ts[1:]:
        if t>30:
            u[...] = 0
        xs_rect.append(A_step @ rectify(xs_rect[-1] + u))
    xs_rect = np.array(xs_rect)
    plt.figure(figsize=(3,2), dpi=400)
    for i in range(10):
        plt.plot(ts, xs_rect[:,i], ls=['-', ':'][0], color=c[i], alpha=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Activity (a.u.)')
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.xlim([-1, 50])
    plt.yticks([0,0.5])
    plt.ylim(-0.1, 0.6)
    plt.legend([f'{(a*180/np.pi):.1f}' for a in np.linspace(0,np.pi, 9)], loc='upper right', bbox_to_anchor=(1.8, 1),
    fontsize=7, ncol=1, columnspacing=0.5, title='Degrees visual angle', title_fontsize=7
    )
    plt.savefig('ring_neuron_stim_' + str(stim_neur)+ '.pdf', bbox_inches='tight')


# %%

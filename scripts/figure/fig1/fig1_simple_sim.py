#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib as mpl
#set the matplotlib font to arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
cmap = 'bwr'
# %% now lets consider correlated noise
np.random.seed(1)
N = 2
W = np.eye(N)
eig = np.linalg.svd(W)[1]
W = W/np.abs(eig[0])/2
T = 200
noise_scale = 1.0
noise = np.random.randn(T, N) * noise_scale
#add several sine waves of random phase
period = [2, 3, 5, 7,10, 16 ]
t = np.arange(T)
m_error = np.array([np.cos((t/T*2*np.pi + np.random.uniform(0,np.pi*2))*P) for P in period]).T
m_error = m_error.sum(1, keepdims=True)
m_error/= m_error.std()
L = np.random.randn(T)
#%%
R = np.zeros((N, T))
R_w_out_m_error = np.zeros((N, T))
R[:, 0] = noise[0, :]
for t in range(1, T):
    R[:, t] = W @ R[:, t-1] + noise[t, :] + m_error[t, :]
    R[0, t] += L[t]
    R_w_out_m_error[:, t] = W @ R_w_out_m_error[:, t-1] + noise[t, :]

#%% make stacked subplot of m_error, R_w_out_m_error, and R
s = 0.8
T_samps = 100
fig, axs = plt.subplots(3,1, figsize=(3*s,4*s), sharey=False, sharex=True, dpi=200)
colors = ['C2', 'C6', 'C5',]
axs[0].plot(m_error[:T_samps], c=colors[0])
axs[0].set_title('Common noise (Z)')
axs[0].set_xticklabels([])

axs[1].plot(R_w_out_m_error[0][:T_samps], c=colors[1])
axs[1].plot(R_w_out_m_error[1][:T_samps], c=colors[2])
axs[1].set_title('Raw responses')

axs[1].set_xticklabels([])

axs[2].plot(R[0][:T_samps], c=colors[1])
axs[2].plot(R[1][:T_samps], c=colors[2])
axs[2].set_title('Confounded responses')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Neural response')
#axs[2].legend(['X', 'Y'], loc=(1.1,0))
axs[0].set_yticklabels([])
axs[2].set_xticks([])
for i in range(3):
    axs[i].set_yticks([])

plt.tight_layout()
plt.savefig('fig1_sim_example.pdf',bbox_inches='tight', transparent=True, pad_inches=0)

#%%

#make logspace int for T from 1000, to 10000
Ts = np.logspace(np.log10(1000), np.log10(100000), 5).astype(int)
period = [2, 3, 5, 7, 10, 16 ]

print(Ts)
n_sims = 100
all_res = []
for T in tqdm(Ts):
    res = []
    for sim in (range(n_sims)):   
        noise = np.random.randn(T, N) * noise_scale
        #add several sine waves of random phase
        t = np.arange(T)
        m_error = np.array([np.cos((t/T*2*np.pi + np.random.uniform(0, np.pi*2))*P) for P in period]).T
        m_error = m_error.sum(1, keepdims=True)
        m_error/= m_error.std()
        L = np.random.randn(T)

        R = np.zeros((N, T))
        R_w_out_m_error = np.zeros((N, T))
        R[:, 0] = noise[0, :]
        for t in range(1, T):
            R[:, t] = W @ R[:, t-1] + noise[t, :] + m_error[t, :]
            R[0, t] += L[t]


        #now do it to get square beta
        hat_W_xy_lstq = (np.linalg.pinv(R[:, :-1] @ R[:, :-1].T) @ R[:, :-1] @ R[:, 1:].T).T

        n_obs = None
        X = R[0,:].T
        Y = R[:, 1:].T
        # calcuate covariance matrix between L[t] and R[t, :n_x]
        hat_W_lx = X.T @ L.T/T
        # calculate covariance matrix between L[t] and R[t+1, n_x:n_x+n_y] using the formula
        cov_lxy = R[:, 1:] @ L[:-1].T/T
        hat_W_lx_pinv = 1/(hat_W_lx)
        #multiply cov_L_y by the pseudoinverse of hat_W_lx to get the estimated W_xy
        _ = cov_lxy * hat_W_lx_pinv
        #fill in other half of matrix with nans
        hat_W_xy_IV = np.zeros((N, N))
        hat_W_xy_IV[...] = np.nan
        hat_W_xy_IV[:, 0] = _

        res.append([hat_W_xy_lstq[1,0], hat_W_xy_IV[1,0]])
    all_res.append(res)
# %%
s=0.6
plt.figure(figsize=(3*s,3*s), dpi=300)
res = np.array(all_res)
color = ['C0', 'C1']
for i in range(2):
    plt.errorbar(Ts, res[:,:,i].mean(1), yerr=res[:,:,i].std(1), label=['Least-squares', 'IV'][i], c=color[i])
plt.semilogx()
#truth in dashed black
plt.plot([Ts[0], Ts[-1]], [0,0], c='k', ls='--', label='Ground\ntruth', zorder=1000)
plt.xlabel('# time samples')
plt.ylabel('Estimate')
#plt.legend(loc=(1.05,0), fontsize=8)
plt.xlim(5e2,2e5)
plt.title('Effect of X on Y')
#replace scientific notation with regular in xticks
xticks = np.array([1e3, 1e4, 1e5]).astype(int)
#add labels with commas
xtick_labels = [f'{x:,}' for x in xticks]
plt.xticks(xticks, xtick_labels)
plt.savefig('IV_vs_lstq_simple.pdf', bbox_inches='tight', transparent=True, pad_inches=0)

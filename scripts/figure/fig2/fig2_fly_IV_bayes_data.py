#%%
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from tqdm import tqdm
import xarray as xr
import scipy as sp
def suff_stats_fit_prior(X, Y, L):
    #generate statistics from which IV and IV-bayes can be calculated
    # regress X on L to give hat_X (just a function of laser) for 1st stage of 2 stage least squares
    hat_W_lx = np.linalg.lstsq(L.T, X.T, rcond=None)[0].T
    hat_X = hat_W_lx @ L 
    hat_W_xy_IV = np.linalg.lstsq(hat_X[:, :-1].T, Y[:,1:].T, rcond=None)[0].T#raw IV estimate
    sig2 = np.var(Y[:, 1:] - hat_W_xy_IV@hat_X[:, :-1], axis=1)#calculate residual variance for IV-bayes
    #see eq 5 in paper for where these are used.
    XTY = np.matmul(hat_X[:, :-1], Y[:, 1:, None]).squeeze(-1)
    XTX = hat_X @ hat_X.T
    return XTX, XTY, sig2, hat_W_xy_IV

def fit_prior_w_suff_stat(XTX, XTY, sig2, a_C, prior_mean_scale=1, prior_var_scale=1):
    #function to take stats from suff_stats_fit_prior and estimate W_xy using prior
    N_stim_neurons = XTX.shape[0]
    prior_W = a_C*prior_mean_scale
    gamma2 = (np.abs(prior_W)+ 1e-16)/prior_var_scale#set prior proportional to connectome
    #eq 5 in paper
    inv_sig2 = 1/sig2
    inv_gamma2 = 1/gamma2
    inv_sig2_XTX = XTX[None, :, :] * inv_sig2[:, None, None]
    inv = np.linalg.inv(inv_sig2_XTX + inv_gamma2[..., None]*np.eye(N_stim_neurons)[None, :, :])
    inv_sig2_XTY = XTY * inv_sig2[:, None, ]
    cov = inv_sig2_XTY + prior_W * inv_gamma2
    hat_W_xy_IV_bayes = np.matmul(inv, cov[..., None]).squeeze()

    return hat_W_xy_IV_bayes

def simulate_LDS(W, source_neuron, T=1000, n_l=1, laser_power=1, seed=None):
    # W is effectome matrix
    # source_neuron is a list of neuron indices that are stimulated
    # T is number of time samples to simulate
    # n_l is number of lasers
    # seed is random seed
    if seed is not None:
        np.random.seed(seed)
    n = W.shape[0]
    W_lx = np.zeros((n, n_l))#matrix that maps lasers to neurons
    W_lx[source_neuron, :] = 1. #laser power is 1 for stimulated neurons
    W_lx = csr_matrix(W_lx)
    L = np.random.randn(n_l, T)*laser_power# lasers, known b/c we control them
    #preallocate R the response matrix
    R = np.zeros((n, T))
    R[:, 0] = np.random.randn(n)#initialize with noise
    #run LDS
    for t in (range(1, T)):
        # weight prior time step, add laser, add noise
        noise = np.random.randn(n)
        R[:, t] = W @ R[:, t-1] + W_lx @ L[:, t] + noise
    return R, L

def sim_connectome(n, syn_count_sgn, post_root_id_unique, pre_root_id_unique, scale_var=1):
    #function to simulate connectome from prior distribution (almost, 0s stay 0, rescaled by largest eigenvalue)) 
    syn_count_sgn_sim = np.random.normal(loc=syn_count_sgn, 
                            scale=np.abs(syn_count_sgn*scale_var)**0.5, 
                            size=syn_count_sgn.shape)

    C_sim = csr_matrix((syn_count_sgn_sim, (post_root_id_unique, pre_root_id_unique)), 
                            shape=(n, n), dtype='float64')
    eigenvalues, _ = eigs(C_sim, k=1)
    scale_sim = 1/np.abs(eigenvalues[0])
    W_sim = C_sim*scale_sim#scale connectome by largest eigenvalue so that it is stable
    return W_sim, C_sim, scale_sim


#%%
top_dir = '../../../data/'
df_sgn = pd.read_csv(top_dir + 'connectome_sgn_cnt.csv', index_col=0)
syn_count_sgn = df_sgn['syn_count_sgn'].values
post_root_id_unique = df_sgn['post_root_id_unique'].values
pre_root_id_unique = df_sgn['pre_root_id_unique'].values

C_orig = sp.sparse.load_npz(top_dir + 'connectome_sgn_cnt.npz')

df_meta_W = pd.read_csv(top_dir + 'meta_data.csv')
eigenvalues = np.load(top_dir + 'eigenvalues_1000.npy')
eig_vec = np.load(top_dir + 'eigvec_1000.npy')
scale_orig = 1/np.abs(eigenvalues[0])

conv_rev = pd.read_csv(top_dir + 'C_index_to_rootid.csv')
conv_dict_rev = dict(zip(conv_rev.iloc[:,0].values, conv_rev.iloc[:,1].values,))

source_neuron = np.array([68521])
n = C_orig.shape[0]
#%%
n_sims = 5
a_C = C_orig[:, source_neuron].toarray()
T = 1000
T_subs = list(np.arange(5, 50, 5)) + list(np.arange(50, 100, 10)) + list(np.arange(100, 1000, 100)) +  list(np.arange(1000, 11000, 1000))
T_subs = [10, 50,  100, 500, 1000, 5000, ]
laser_powers = [10.,]

#data array for suff stats of target neurons
dims = ['w', 'sim', 'T_sub', 'est', 'laser_power']
coords = {'w':range(len(a_C)), 'sim':range(n_sims), 'T_sub':T_subs, 
                'est':['XTY', 'sig2', 'IV'], 'laser_power':laser_powers}
da_target = xr.DataArray(np.zeros((len(a_C), n_sims, len(T_subs), 3, len(laser_powers))), 
                        dims=dims, coords=coords)

#data array for suff stats of source neurons
dims = ['sim', 'T_sub', 'laser_power']
coords = {'sim':range(n_sims), 'T_sub':T_subs, 'laser_power':laser_powers}
da_source = xr.DataArray(np.zeros((n_sims, len(T_subs), len(laser_powers))), dims=dims, coords=coords)

#data array for true weights
dims = ['w', 'sim']
coords = {'w':range(len(a_C)), 'sim':range(n_sims), }
w_true = xr.DataArray(np.zeros((len(a_C), n_sims)), 
        dims=dims, coords=coords)

for laser_power in (laser_powers):
    for T_sub in tqdm(T_subs):
        for sim in range(n_sims):
            W, C, scale = sim_connectome(n, syn_count_sgn, post_root_id_unique, pre_root_id_unique, scale_var=1)
            R, L = simulate_LDS(W, source_neuron, T=T_sub, n_l=1, laser_power=laser_power,
                        seed=None)
            w_true[:, sim] = W[:, source_neuron].toarray().squeeze()
            
            X = R[source_neuron, :T_sub]
            Y = R[:, :T_sub]
            a_L = L[:, :T_sub]
            XTX, XTY, sig2, hat_W_xy_IV = suff_stats_fit_prior(X, Y, a_L)
            da_target.loc[dict(sim=sim, T_sub=T_sub, est='XTY', laser_power=laser_power)] = XTY.squeeze()
            da_target.loc[dict(sim=sim, T_sub=T_sub, est='sig2', laser_power=laser_power)] = sig2.squeeze()
            da_target.loc[dict(sim=sim, T_sub=T_sub, est='IV', laser_power=laser_power)] = hat_W_xy_IV.squeeze()
            da_source.loc[dict(sim=sim, T_sub=T_sub, laser_power=laser_power)] = XTX.squeeze()
    ds = xr.Dataset({'target':da_target, 'source':da_source, 'w_true':w_true})
# %%
dims = ['w', 'sim', 'T_sub', 'est', 'laser_power']
coords = {'w':range(len(a_C)), 'sim':range(n_sims), 'T_sub':T_subs, 'est':['bayes', 'IV'], 'laser_power':laser_powers}
da_est = xr.DataArray(np.zeros((len(a_C), n_sims, len(T_subs), 2, len(laser_powers))), dims=dims, coords=coords)

#now use suff stats to get estimate
for laser_power in laser_powers:
    for sim in tqdm(range(n_sims)):
        for T_sub in (T_subs):
            XTX = ds['source'].sel(sim=sim, T_sub=T_sub, laser_power=laser_power).values[None,None]
            XTY = ds['target'].sel(sim=sim, T_sub=T_sub, est='XTY', laser_power=laser_power).values[:,None]
            sig2 = ds['target'].sel(sim=sim, T_sub=T_sub, est='sig2', laser_power=laser_power).values
            hat_W_xy_IV_bayes = fit_prior_w_suff_stat(XTX, XTY, sig2, a_C, prior_mean_scale=scale_orig)
            da_est.loc[dict(sim=sim, T_sub=T_sub, est='bayes', laser_power=laser_power)] = hat_W_xy_IV_bayes.squeeze()
            da_est.loc[dict(sim=sim, T_sub=T_sub, est='IV', laser_power=laser_power)] = ds['target'].sel(sim=sim, T_sub=T_sub, est='IV', laser_power=laser_power).values.squeeze()
#%%
#save all results as netcdf
top_dir = '../../data/'
ds.to_netcdf(top_dir + '/sim_results.nc')
da_est.to_netcdf( top_dir + '/sim_estimates.nc')

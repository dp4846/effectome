#%% 
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import numpy as np
import scipy as sp

neurotransmitter_effects = {
    'ACH': 1,    # Excitatory
    'DA': 1,     # Excitatory (can have inhibitory effects in specific brain regions)
    'GABA': -1,  # Inhibitory
    'GLUT': -1,   # Inhibitory (but excitatory in mammals!)
    'OCT': -1,   # Inhibitory
    'SER': -1    # Inhibitory (can have excitatory effects in specific brain circuits)
    }

# load up dynamics matrix and meta data (get this from https://codex.flywire.ai/api/download)
top_dir = '../../data/'
fn = top_dir + 'connections.csv'
df = pd.read_csv(fn)#read in
fn = top_dir + 'classification.csv'
df_meta = pd.read_csv(fn)#read in
#%%
df['nt_sign'] = df['nt_type'].map(neurotransmitter_effects)#convert type to sign in the data frame
df['syn_cnt_sgn'] = df['syn_count']*df['nt_sign']#multiply count by sign for unscaled 'effectome'
df = df.groupby(['pre_root_id', 'post_root_id']).agg({'syn_cnt_sgn': 'sum', 'syn_count': 'sum', 'neuropil':'first', 
                                                            'nt_type':'first'}).reset_index()#sum synapses across all unique pre and post pairs
vals, inds, inv = np.unique(list(df['pre_root_id'].values) + list(df['post_root_id'].values), return_index=True, return_inverse=True)
conv_dict = {val:i for i, val in enumerate(vals)}
df['pre_root_id_unique'] = [conv_dict[val] for val in df['pre_root_id']]
df['post_root_id_unique'] = [conv_dict[val] for val in df['post_root_id']]
n = len(vals)#total rows of full dynamics matrix
syn_count = df['syn_count'].values
is_syn = (syn_count>0).astype('int')
syn_count_sgn = df['syn_cnt_sgn'].values
pre_root_id_unique = df['pre_root_id_unique'].values
post_root_id_unique = df['post_root_id_unique'].values
# form unscaled sparse matrix
k_eig_vecs = 1000#number of eigenvectors to use
C_orig = csr_matrix((syn_count_sgn, (post_root_id_unique, pre_root_id_unique)), shape=(n, n), dtype='float64')
eigenvalues, eig_vec = eigs(C_orig, k=k_eig_vecs)#get eigenvectors and values (only need first for scaling)
scale_orig = 0.99/np.abs(eigenvalues[0])#make just below stability
W_full = C_orig*scale_orig#scale connectome by largest eigenvalue so that it decays
# dictionary to go back to original ids of cells
conv_dict_rev = {v:k for k, v in conv_dict.items()}
root_id = [conv_dict_rev[i] for i in range(n)]
df_meta_W = df_meta.set_index('root_id', ).loc[root_id]
df_meta_W['root_id_W'] = np.arange(n)
df_meta_W = df_meta_W.set_index('root_id_W')
df_meta_W['root_id'] = root_id

df_sgn = pd.DataFrame({'syn_count_sgn':syn_count_sgn, 'pre_root_id_unique':pre_root_id_unique, 'post_root_id_unique':post_root_id_unique})
df_sgn.to_csv(top_dir + 'connectome_sgn_cnt.csv')
sp.sparse.save_npz(top_dir + 'connectome_sgn_cnt.npz', C_orig)

np.save(top_dir + 'eigenvalues_' + str(k_eig_vecs) + '.npy', eigenvalues)
np.save(top_dir + 'eigvec_' + str(k_eig_vecs) + '.npy', eig_vec)

df_meta_W.to_csv(top_dir + 'meta_data.csv')

df_conv = pd.DataFrame.from_dict(conv_dict_rev, orient='index')
df_conv.to_csv(top_dir + 'C_index_to_rootid.csv')
# %%

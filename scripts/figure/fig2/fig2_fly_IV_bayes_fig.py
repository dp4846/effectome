#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib as mpl
#set the matplotlib font to arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
top_dir = '../../data/'
ds = xr.open_dataset(top_dir + '/sim_results.nc')#load simulations from _data
da_est= xr.open_dataarray( top_dir + '../data/sim_estimates.nc')

#%% example scatters of IV and Bayes-IV
s = 2
sim = 1
lim = 0.04
ticks = np.linspace(-lim, lim, 5)
colors = ['C0', 'C1',]
titles = ['Bayes-IV', 'IV']
for i, est in enumerate(['bayes', 'IV']):
    plt.figure(figsize=(s,s), dpi=300)
    y = da_est.sel(sim=sim, T_sub=1000, est=est, laser_power=10).values
    plt.scatter(ds['w_true'][:, sim], y, s=5, alpha=0.9, color=colors[i], rasterized=True)
    plt.gca().set_xlim(-lim, lim)
    plt.gca().set_ylim(-lim, lim)
    plt.xticks(ticks)
    plt.yticks(ticks)
    #set tick labels just to be ends and middle
    tick_labels = [str(ticks[0]), '', '0', '', str(ticks[-1])]
    plt.gca().set_xticklabels(tick_labels)
    plt.gca().set_yticklabels(tick_labels)
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('True weights')
    plt.ylabel('Estimated weights')
    #make a diagonal line
    plt.plot([-lim, lim], [-lim, lim], color='k', linewidth=0.5)
    plt.title(titles[i])
    plt.savefig('wholebrain_example_' +  est + '.pdf', bbox_inches='tight', transparent=True)
#%%
#now estimate r2
da_true = ds['w_true']
rss = ((da_true - da_est)**2).sum('w')
tss = (da_true**2).sum('w')
r2 = 1 - rss/tss
rss_mean = rss.mean('sim')
rss_std = rss.std('sim')
x = rss_mean.coords['T_sub']
fig, ax = plt.subplots(1,1, figsize=(2,2), sharex=False, sharey=True)
est_types = ['IV', 'bayes']
tss_av = tss.mean()
colors = ['C0', 'C1']
xticks = [10, 100, 1000, 10000]
ax.loglog(x, rss_mean.sel(est='IV',), label='IV', color=colors[0])
ax.loglog(x, rss_mean.sel(est='bayes',), label='IV-bayes', color=colors[1])
ax.axhline(tss_av, color='k', linestyle='--', label=r'$R^2=0$')
ax.axhline(tss_av/10, color='k', linestyle=':', label=r'$R^2=0.9$')
ax.set_xticks(xticks)
ax.legend(loc=(1.1, 0))
ax.set_ylabel(r'RSS')
ax.set_xlabel(r'# samples')
#make l
plt.savefig('wholebrain_efficiency.svg', bbox_inches='tight', transparent=True)
#%% prior distribution example
mu = [0, 8, 0, -4, 0, 0, 0, 0, 5, 0]
sd = np.array(np.abs(mu)) + 1
x = np.arange(10)
plt.figure(figsize=(2,2), dpi=300)
plt.errorbar(x, mu, yerr=sd, fmt='o', color='k', capsize=2, capthick=1, elinewidth=1, markersize=1)
plt.grid()
plt.xlabel('Post-synaptic neuron index')
plt.ylabel('Prior distribution')
#make symmetric across y axis
plt.gca().set_ylim(-20, 20)
plt.savefig('wholebrain_prior.pdf', bbox_inches='tight', transparent=True)
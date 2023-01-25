#!/usr/bin/env python3

import os
import argparse
import re

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib as mpl
from mycolorpy import colorlist as mcp
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import colors
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

def create_dataframe(directory, stat, metric, **kwargs):
    os.chdir(directory)
    path = os.getcwd()
    df_out = pd.DataFrame()
    #loops over different dimensions
    for dimdir in os.listdir(path):
        dim_path = os.path.join(path, dimdir)
        #loops over different tests
        distribution = []
        for testdir in os.listdir(dim_path):
            df = pd.read_csv(os.path.join(dim_path, testdir, 'results.csv'))
            distribution.extend(df.loc[df['desc'] == stat, metric].to_numpy())
        df_out[dimdir] = distribution

    return df_out

def create_dataframe2(directory, test_config: list, stat, metric, **kwargs):
    binary = test_config['binary']
    nproc = test_config['num_proc']
    dims  = test_config['dimensions']
    nruns = test_config['num_runs']
    os.chdir(directory)
    path = os.getcwd()
    df_out = pd.DataFrame()
    #loops over different dimensions
    for dim in dims:
        #loops over different tests
        distribution = []
        j = 0
        for file in os.listdir(path):
            filename = f"{binary}_n{nproc}_s{dim}_r{j}.csv"
            if (file == filename):
                filepath = os.path.join(path, filename)
                df = pd.read_csv(filepath)
                distribution.extend(df.loc[df['desc'] == stat, metric].to_numpy())
                j=j+1
        df_loop=pd.DataFrame(distribution)
        df_out = pd.concat([df_out, df_loop], axis=1, ignore_index=1)
    return df_out

def mean_confidence_interval(data, confidence):
    m = data.apply(np.mean, axis=0)
    se = data.apply(stats.sem, nan_policy='omit')
    n = len(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, se, h

def mean_confidence_interval_ratio(data1, data2, confidence):
    m=[]
    se=[]
    h=[]
    for column in data1:
        div=[]
        for i in range(0, len(data1[column])):
            for j in range(0, len(data2[column])):
                x = data1.loc[i].at[column]/data2.loc[j].at[column]
                div.append(x)
        n = len(div)
        m.append(np.nanmean(div))
        se.append(stats.sem(div, nan_policy='omit'))
        h.append(stats.sem(div, nan_policy='omit') * stats.t.ppf((1 + confidence) / 2., n-1))
    return m, se, h

directory = '/usr/scratch2/larain9/yiczhang/proj/DPHPC/snitch/sw/dphpc/results'
dims = np.array([4, 8, 16]);
dims_label = ['In4x4-Ch8', 'In8x8-Ch8', 'In16x16-Ch8'];
dims_normalized = np.log2(dims)*4

# DDD CONV2D - SINGLE
config = {'binary': 'conv2d_dense_dense_dense', 'num_proc': 1, 'dimensions': dims, 'num_runs': 1}
sc_ddd_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
sc_ddd_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
sc_ddd_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')

# DDD CONV2D - PARALLEL
config = {'binary': 'conv2d_dense_dense_dense', 'num_proc': 8, 'dimensions': dims, 'num_runs': 10}
mc_ddd_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
mc_ddd_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
mc_ddd_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
mc_ddd_synchov = create_dataframe2(directory, config, 'mean', 'synch_overhead')

# CCD CONV2D - SINGLE
config = {'binary': 'conv2d_csr_csr_dense', 'num_proc': 1, 'dimensions': dims, 'num_runs': 10}
sc_ccd_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
sc_ccd_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
sc_ccd_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')

# CCD CONV2D - PARALLEL
config = {'binary': 'conv2d_csr_csr_dense', 'num_proc': 8, 'dimensions': dims, 'num_runs': 10}
mc_ccd_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
mc_ccd_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
mc_ccd_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
mc_ccd_synchov = create_dataframe2(directory, config, 'mean', 'synch_overhead')

# CDD CONV2D - SINGLE
config = {'binary': 'conv2d_csr_dense_dense', 'num_proc': 1, 'dimensions': dims, 'num_runs': 10}
sc_cdd_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
sc_cdd_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
sc_cdd_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')

# CDD CONV2D - PARALLEL
config = {'binary': 'conv2d_csr_dense_dense', 'num_proc': 8, 'dimensions': dims, 'num_runs': 10}
mc_cdd_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
mc_cdd_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
mc_cdd_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
mc_cdd_synchov = create_dataframe2(directory, config, 'mean', 'synch_overhead')

# DCD CONV2D - SINGLE
config = {'binary': 'conv2d_dense_csr_dense', 'num_proc': 1, 'dimensions': dims, 'num_runs': 10}
sc_dcd_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
sc_dcd_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
sc_dcd_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')

# DCD CONV2D - PARALLEL
config = {'binary': 'conv2d_dense_csr_dense', 'num_proc': 8, 'dimensions': dims, 'num_runs': 10}
mc_dcd_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
mc_dcd_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
mc_dcd_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
mc_dcd_synchov = create_dataframe2(directory, config, 'mean', 'synch_overhead')

# DCD_CSRR CONV2D - SINGLE
config = {'binary': 'conv2d_dense_csrr_dense', 'num_proc': 1, 'dimensions': dims, 'num_runs': 10}
sc_dcrd_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
sc_dcrd_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
sc_dcrd_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')

# DCD CONV2D - PARALLEL
config = {'binary': 'conv2d_dense_csrr_dense', 'num_proc': 8, 'dimensions': dims, 'num_runs': 10}
mc_dcrd_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
mc_dcrd_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
mc_dcrd_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
mc_dcrd_synchov = create_dataframe2(directory, config, 'mean', 'synch_overhead')

#############################################################################
# PLOT FEATURES

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 22

mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
mpl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
mpl.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
fe = font_manager.FontEntry(
    fname='/home/mbertuletti/.local/share/fonts/helvetica/Helvetica.otf',
    name='FreeSans'
)
font_manager.fontManager.ttflist.insert(0, fe)
mpl.rcParams['font.family'] = fe.name


#############################################################################
# CYCLES PLOT

# plot cycles:
fig0 = plt.figure(figsize=(15, 6))
ax0= plt.subplot2grid((2, 2), (0, 0), rowspan=1)
ax1= plt.subplot2grid((2, 2), (1, 0), rowspan=1)
ax2= plt.subplot2grid((2, 2), (0, 1), rowspan=2)
cmap = mcp.gen_color(cmap="Spectral_r",n=6)

##LINES
#m, se, h = mean_confidence_interval(sc_dense_cycles, 0.95)
#l1, = ax0.plot(dims, m, marker="o", linewidth=2.0, color=cmap[0])
#m, se, h = mean_confidence_interval(sc_csr0_cycles, 0.95)
#l2, = ax0.plot(dims, m, marker="x", linewidth=2.0, color=cmap[1])
#plt.fill_between(dims, m - h, m + h, color='k', alpha=0.2)
#m, se, h = mean_confidence_interval(sc_csr1_cycles, 0.95)
#l3, = ax0.plot(dims, m, marker="x", linewidth=2.0, color=cmap[2])
#plt.fill_between(dims, m - h, m + h, color='k', alpha=0.2)
#BARS
width = 0.66
m, se, h = mean_confidence_interval(sc_ddd_cycles, 0.95)
l1 = ax0.bar(dims_normalized-4*width/2, m, width, color=cmap[0], edgecolor='k')

m, se, h = mean_confidence_interval(sc_ccd_cycles, 0.95)
l2 = ax0.bar(dims_normalized-2*width/2, m, width, color=cmap[1], edgecolor='k')
ax0.errorbar(dims_normalized-2*width/2, m, 4*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval(sc_cdd_cycles, 0.95)
l3 = ax0.bar(dims_normalized, m, width, color=cmap[2], edgecolor='k')
ax0.errorbar(dims_normalized, m, 4*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval(sc_dcd_cycles, 0.95)
l4 = ax0.bar(dims_normalized+2*width/2, m, width, color=cmap[3], edgecolor='k')
ax0.errorbar(dims_normalized+2*width/2, m, 4*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval(sc_dcrd_cycles, 0.95)
l5 = ax0.bar(dims_normalized+4*width/2, m, width, color=cmap[4], edgecolor='k')
ax0.errorbar(dims_normalized+4*width/2, m, 4*np.array(h), fmt='none', ecolor='r', elinewidth=2)

ax0.legend([l1, l2, l3, l4, l5], ['Dense-Dense-Dense', 'CSR-CSR-DENSE', 'CSR-DENSE-DENSE', 'DENSE-CSR-DENSE', 'DENSE-CSRv2-DENSE'], loc='upper left', facecolor='white', framealpha=1, fontsize=SMALL_SIZE)
ax0.set_xticks(dims_normalized, dims_label)
ax0.set_xlabel('Input dimension')
ax0.set_ylabel('Cycles')
ax0.set_title('CONV2D Single-core')
ax0.set(ylim=(6*10**3, 2*10**6), yticks=np.arange(6*10**3, 2*10**6, 10**4), yscale="log")
#ax0.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)

ax0.grid(True)
plt.tight_layout()


##LINES
#m, se, h = mean_confidence_interval(mc_dense_cycles, 0.95)
#l1, = ax1.plot(dims_normalized, m, marker="o", linewidth=2.0, color=cmap[0])
#m, se, h = mean_confidence_interval(mc_csr0_cycles, 0.95)
#l2, = ax1.plot(dims_normalized, m, marker="x", linewidth=2.0, color=cmap[1])
#plt.fill_between(dims_normalized, m - h, m + h, color='k', alpha=0.2)
#m, se, h = mean_confidence_interval(mc_csr1_cycles, 0.95)
#l3, = ax1.plot(dims_normalized, m, marker="x", linewidth=2.0, color=cmap[2])
#plt.fill_between(dims_normalized, m - h, m + h, color='k', alpha=0.2)
#BARS

m, se, h = mean_confidence_interval(mc_ddd_cycles, 0.95)
l1 = ax1.bar(dims_normalized-4*width/2, m, width, color=cmap[0], edgecolor='k')

m, se, h = mean_confidence_interval(mc_ccd_cycles, 0.95)
l2 = ax1.bar(dims_normalized-2*width/2, m, width, color=cmap[1], edgecolor='k')
ax1.errorbar(dims_normalized-2*width/2, m, 4*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval(mc_cdd_cycles, 0.95)
l3 = ax1.bar(dims_normalized, m, width, color=cmap[2], edgecolor='k')
ax1.errorbar(dims_normalized, m, 4*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval(mc_dcd_cycles, 0.95)
l4 = ax1.bar(dims_normalized+2*width/2, m, width, color=cmap[3], edgecolor='k')
ax1.errorbar(dims_normalized+2*width/2, m, 4*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval(mc_dcrd_cycles, 0.95)
l5 = ax1.bar(dims_normalized+4*width/2, m, width, color=cmap[4], edgecolor='k')
ax1.errorbar(dims_normalized+4*width/2, m, 4*np.array(h), fmt='none', ecolor='r', elinewidth=2)

ax1.legend([l1, l2, l3, l4, l5], ['Dense-Dense-Dense', 'CSR-CSR-DENSE', 'CSR-DENSE-DENSE', 'DENSE-CSR-DENSE', 'DENSE-CSRv2-DENSE'], loc='upper left', facecolor='white', framealpha=1, fontsize=SMALL_SIZE)
ax1.set_xticks(dims_normalized, dims_label)
ax1.set_xlabel('Input dimension')
ax1.set_ylabel('Cycles')
ax1.set_title('CONV2D 8-cores')
ax1.set(ylim=(8*10**2, 3*10**5), yticks=np.arange(8*10**2, 3*10**5, 10**3), yscale="log")
#ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
ax1.grid(True)
plt.tight_layout()


##LINES
#m, se, h = mean_confidence_interval_ratio(sc_dense_cycles, mc_dense_cycles, 0.95)
#l1, = ax2.plot(dims_normalized, m, marker="o", linewidth=2.0, color=cmap[0])
#m, se, h = mean_confidence_interval_ratio(sc_csr0_cycles, mc_csr0_cycles, 0.95)
#l2, = ax2.plot(dims_normalized, m, marker="x", linewidth=2.0, color=cmap[1])
#plt.fill_between(dims_normalized, np.array(m) - np.array(h), m + np.array(h), color='k', alpha=0.2)
#m, se, h = mean_confidence_interval_ratio(sc_csr1_cycles, mc_csr1_cycles, 0.95)
#l3, = ax2.plot(dims_normalized, m, marker="x", linewidth=2.0, color=cmap[2])
#plt.fill_between(dims_normalized, np.array(m) - np.array(h), np.array(m) + np.array(h), color='k', alpha=0.2)
#BARS
m, se, h = mean_confidence_interval_ratio(sc_ddd_cycles, mc_ddd_cycles, 0.95)
l1 = ax2.bar(dims_normalized-4*width/2, m, width, color=cmap[0], edgecolor='k')

m, se, h = mean_confidence_interval_ratio(sc_ccd_cycles, mc_ccd_cycles, 0.95)
l2 = ax2.bar(dims_normalized-2*width/2, m, width, color=cmap[1], edgecolor='k')
ax2.errorbar(dims_normalized-2*width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval_ratio(sc_cdd_cycles, mc_cdd_cycles, 0.95)
l3 = ax2.bar(dims_normalized, m, width, color=cmap[2], edgecolor='k')
ax2.errorbar(dims_normalized, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval_ratio(sc_dcd_cycles, mc_dcd_cycles, 0.95)
l4 = ax2.bar(dims_normalized+2*width/2, m, width, color=cmap[3], edgecolor='k')
ax2.errorbar(dims_normalized+2*width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval_ratio(sc_dcrd_cycles, mc_dcrd_cycles, 0.95)
l5 = ax2.bar(dims_normalized+4*width/2, m, width, color=cmap[4], edgecolor='k')
ax2.errorbar(dims_normalized+4*width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

ax2.legend([l1, l2, l3, l4, l5], ['Dense-Dense-Dense', 'CSR-CSR-DENSE', 'CSR-DENSE-DENSE', 'DENSE-CSR-DENSE', 'DENSE-CSRv2-DENSE'], loc='upper right', facecolor='white', framealpha=1, fontsize=SMALL_SIZE)
ax2.set_xticks(dims_normalized, dims)
ax2.set_xlabel('Input dimension')
ax2.set_ylabel('Speed-up')
ax2.set_title('CONV2D Speed-UP')
ax2.set(ylim=(0,10.8), yticks=np.arange(0, 10.8, 1))
ax2.text(6.7, 8.2, 'IDEAL Speed-UP', color='r', fontsize=MEDIUM_SIZE)
ax2.axhline(y = 8, color = 'r', linestyle = '--')
ax2.grid(True)
plt.tight_layout()


#############################################################################
# IPC PLOT
width = 0.73
fig1 = plt.figure(figsize=(11, 6))
ax0= plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax1= plt.subplot2grid((2, 2), (1, 0), colspan=2)
cmap = mcp.gen_color(cmap="RdBu",n=6)
cmap=cmap[2:4]
patterns = [ "", "/" , "o", "x", "." ]

ax0.bar(dims_normalized-6*width/3, sc_ddd_coreipc.apply(np.mean, axis=0), width,
        color=cmap[0], hatch=patterns[0], edgecolor='k')
ax0.bar(dims_normalized-6*width/3, sc_ddd_fpssipc.apply(np.mean, axis=0), width,
        bottom=sc_ddd_coreipc.apply(np.mean, axis=0),
        color=cmap[1], hatch=patterns[0], edgecolor='k')

ax0.bar(dims_normalized-3*width/3, sc_ccd_coreipc.apply(np.mean, axis=0), width,
        yerr=3*sc_ccd_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax0.bar(dims_normalized-3*width/3, sc_ccd_fpssipc.apply(np.mean, axis=0), width,
        yerr=sc_ccd_fpssipc.apply(np.std, axis=0), bottom=sc_ccd_coreipc.apply(np.mean, axis=0),
        color=cmap[1], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

ax0.bar(dims_normalized, sc_cdd_coreipc.apply(np.mean, axis=0), width,
        yerr=3*sc_cdd_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax0.bar(dims_normalized, sc_cdd_fpssipc.apply(np.mean, axis=0), width,
        yerr=sc_cdd_fpssipc.apply(np.std, axis=0), bottom=sc_cdd_coreipc.apply(np.mean, axis=0),
        color=cmap[1], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

ax0.bar(dims_normalized+3*width/3, sc_dcd_coreipc.apply(np.mean, axis=0), width,
        yerr=3*sc_dcd_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[3], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax0.bar(dims_normalized+3*width/3, sc_dcd_fpssipc.apply(np.mean, axis=0), width,
        yerr=sc_dcd_fpssipc.apply(np.std, axis=0), bottom=sc_dcd_coreipc.apply(np.mean, axis=0),
        color=cmap[1], hatch=patterns[3], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

ax0.bar(dims_normalized+6*width/3, sc_dcrd_coreipc.apply(np.mean, axis=0), width,
        yerr=3*sc_dcrd_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[4], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax0.bar(dims_normalized+6*width/3, sc_dcrd_fpssipc.apply(np.mean, axis=0), width,
        yerr=sc_dcrd_fpssipc.apply(np.std, axis=0), bottom=sc_dcrd_coreipc.apply(np.mean, axis=0),
        color=cmap[1], hatch=patterns[4], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

ax0.legend( [plt.bar([0], [0], color=cmap[0], edgecolor='k'),
             plt.bar([0], [0], color=cmap[1], edgecolor='k'),
             plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[0]),
             plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[1]),
             plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[2]),
             plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[3]),
             plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[4])],
            ["INT-core IPC", "FP-SS IPC", "DENSE-DENSE-DENSE", "CSR-CSR-DENSE", 'CSR-DENSE-DENSE', 'DENSE-CSR-DENSE', 'DENSE-CSRv2-DENSE'],
            loc='upper right', bbox_to_anchor=(1.25, 1.05), facecolor='white', framealpha=1, fontsize=MEDIUM_SIZE)
ax0.set_xticks(dims_normalized, dims_label)
ax0.set(ylim=(0,1.2), yticks=np.arange(0, 1.3, 0.2))
ax0.text(6.7, 0.85, 'IDEAL IPC', color='r', fontsize=MEDIUM_SIZE)
ax0.axhline(y = 1, color = 'r', linestyle = '--')
ax0.set_xlabel('Input dimension')
ax0.set_ylabel('Occupation')
ax0.set_title('Occupation Single-core')
ax0.grid(True)
plt.tight_layout()

s1 = mc_ddd_coreipc.apply(np.mean, axis=0)
s2 = mc_ddd_fpssipc.apply(np.mean, axis=0)
s3 = mc_ddd_synchov.apply(np.mean, axis=0)
ax1.bar(dims_normalized-4*width/2, s1, width, color=cmap[0], hatch=patterns[0], edgecolor='k')
ax1.bar(dims_normalized-4*width/2, s2, width, bottom=s1, color=cmap[1], hatch=patterns[0], edgecolor='k')
ax1.bar(dims_normalized-4*width/2, s3, width, bottom=s1+s2, color='lightgray', edgecolor='k')

s1 = mc_ccd_coreipc.apply(np.mean, axis=0)
s2 = mc_ccd_fpssipc.apply(np.mean, axis=0)
s3 = mc_ccd_synchov.apply(np.mean, axis=0)
ax1.bar(dims_normalized-2*width/2, s1, width, yerr=mc_ccd_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized-2*width/2, s2, width, yerr=mc_ccd_fpssipc.apply(np.std, axis=0), bottom=s1,
        color=cmap[1], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized-2*width/2, s3, width, yerr=mc_ccd_synchov.apply(np.std, axis=0), bottom=s1+s2,
        color='lightgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

s1 = mc_cdd_coreipc.apply(np.mean, axis=0)
s2 = mc_cdd_fpssipc.apply(np.mean, axis=0)
s3 = mc_cdd_synchov.apply(np.mean, axis=0)
ax1.bar(dims_normalized, s1, width, yerr=mc_ccd_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized, s2, width, yerr=mc_ccd_fpssipc.apply(np.std, axis=0), bottom=s1,
        color=cmap[1], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized, s3, width, yerr=mc_ccd_synchov.apply(np.std, axis=0), bottom=s1+s2,
        color='lightgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

s1 = mc_dcd_coreipc.apply(np.mean, axis=0)
s2 = mc_dcd_fpssipc.apply(np.mean, axis=0)
s3 = mc_dcd_synchov.apply(np.mean, axis=0)
ax1.bar(dims_normalized+2*width/2, s1, width, yerr=mc_ccd_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[3], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized+2*width/2, s2, width, yerr=mc_ccd_fpssipc.apply(np.std, axis=0), bottom=s1,
        color=cmap[1], hatch=patterns[3], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized+2*width/2, s3, width, yerr=mc_ccd_synchov.apply(np.std, axis=0), bottom=s1+s2,
        color='lightgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

s1 = mc_dcrd_coreipc.apply(np.mean, axis=0)
s2 = mc_dcrd_fpssipc.apply(np.mean, axis=0)
s3 = mc_dcrd_synchov.apply(np.mean, axis=0)
ax1.bar(dims_normalized+4*width/2, s1, width, yerr=mc_ccd_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[4], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized+4*width/2, s2, width, yerr=mc_ccd_fpssipc.apply(np.std, axis=0), bottom=s1,
        color=cmap[1], hatch=patterns[4], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized+4*width/2, s3, width, yerr=mc_ccd_synchov.apply(np.std, axis=0), bottom=s1+s2,
        color='lightgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

ax1.legend( [plt.bar([3], [0], color=cmap[0], edgecolor='k'),
             plt.bar([3], [0], color=cmap[1], edgecolor='k'),
             plt.bar([3], [0], color='lightgray', edgecolor='k'),
             plt.bar([3], [0], color='w', edgecolor='k'),
             plt.bar([3], [0], color='w', edgecolor='k', hatch=patterns[1]),
             plt.bar([3], [0], color='w', edgecolor='k', hatch=patterns[2]),
             plt.bar([3], [0], color='w', edgecolor='k', hatch=patterns[3]),
             plt.bar([3], [0], color='w', edgecolor='k', hatch=patterns[4])],
            ["INT-core IPC", "FP-SS IPC", "Synch.", "DENSE-DENSE-DENSE", "CSR-CSR-DENSE", 'CSR-DENSE-DENSE', 'DENSE-CSR-DENSE', 'DENSE-CSRv2-DENSE'],
            loc='upper right', bbox_to_anchor=(1.25, 1.05), facecolor='white', framealpha=1, fontsize=MEDIUM_SIZE)
ax1.set_xticks(dims_normalized, dims)
ax1.set(ylim=(0,1.2), yticks=np.arange(0, 1.3, 0.2))
ax1.set(xlim=ax0.get_xlim())
ax1.text(6.7, 0.85, 'IDEAL IPC', color='r', fontsize=MEDIUM_SIZE)
ax1.axhline(y = 1, color = 'r', linestyle = '--')
ax1.set_xlabel('Input dimension')
ax1.set_ylabel('Occupation')
ax1.set_title('Occupation 8-cores')
ax1.grid(True)
plt.tight_layout()

plt.show()

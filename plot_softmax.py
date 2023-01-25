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
    axis  = test_config['axis']
    os.chdir(directory)
    path = os.getcwd()
    df_out = pd.DataFrame()
    #loops over different dimensions
    for dim in dims:
        #loops over different tests
        distribution = []
        j = 0
        for file in os.listdir(path):
            filename = f"{binary}_n{nproc}_s{dim}_r{j}_a{test_config['axis']}.csv"
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
    se = data.apply(stats.sem)
    n = len(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, se, h

def mean_confidence_interval_ratio(data1, data2, confidence):
    m=[]
    se=[]
    h=[]
    for column in data1:
        div=[]
        for item1 in data1[column].items():
            for item2 in data2[column].items():
                if (pd.notna(item2[1])):
                    x = item1[1]/item2[1]
                    div.append(x)
        n = len(div)
        m.append(np.mean(div))
        se.append(stats.sem(div))
        h.append(stats.sem(div) * stats.t.ppf((1 + confidence) / 2., n-1))
    return m, se, h

## DENSE SOFTMAX - SINGLE - AX0
#dims = np.array([10, 20, 30 , 40, 50]);
#directory = '/scratch2/mbertuletti/snitch/results_softmaxdense_sc_axis0'
#sc_dense_cycles = create_dataframe(directory, 'mean', 'cycles')
#sc_dense_coreipc = create_dataframe(directory, 'mean', 'snitch_occupancy')
#sc_dense_fpssipc = create_dataframe(directory, 'mean', 'fpss_occupancy')
## CSR SOFTMAX - SINGLE - AX0
#directory = '/scratch2/mbertuletti/snitch/results_softmax_sc_axis0'
#sc_csr0_cycles = create_dataframe(directory, 'mean', 'cycles')
#sc_csr0_coreipc = create_dataframe(directory, 'mean', 'snitch_occupancy')
#sc_csr0_fpssipc = create_dataframe(directory, 'mean', 'fpss_occupancy')
## CSR SOFTMAX - SINGLE - AX1
#directory = '/scratch2/mbertuletti/snitch/results_softmax_sc_axis1'
#sc_csr1_cycles = create_dataframe(directory, 'mean', 'cycles')
#sc_csr1_coreipc = create_dataframe(directory, 'mean', 'snitch_occupancy')
#sc_csr1_fpssipc = create_dataframe(directory, 'mean', 'fpss_occupancy')
## DENSE SOFTMAX - PARALLEL - AX0
#directory = '/scratch2/mbertuletti/snitch/results_softmaxdense_mc_axis0'
#mc_dense_cycles = create_dataframe(directory, 'mean', 'cycles')
#mc_dense_coreipc = create_dataframe(directory, 'mean', 'snitch_occupancy')
#mc_dense_fpssipc = create_dataframe(directory, 'mean', 'fpss_occupancy')
## CSR SOFTMAX -PARALLEL - AX0
#directory = '/scratch2/mbertuletti/snitch/results_softmax_mc_axis0'
#mc_csr0_cycles = create_dataframe(directory, 'mean', 'cycles')
#mc_csr0_coreipc = create_dataframe(directory, 'mean', 'snitch_occupancy')
#mc_csr0_fpssipc = create_dataframe(directory, 'mean', 'fpss_occupancy')
## CSR SOFTMAX -PARALLEL - AX1
#directory = '/scratch2/mbertuletti/snitch/results_softmax_mc_axis1'
#mc_csr1_cycles = create_dataframe(directory, 'mean', 'cycles')
#mc_csr1_coreipc = create_dataframe(directory, 'mean', 'snitch_occupancy')
#mc_csr1_fpssipc = create_dataframe(directory, 'mean', 'fpss_occupancy')


directory = '/scratch2/mbertuletti/snitch/sw/dphpc/results4'
dims = np.array([8, 16, 32, 64]);
dims_normalized = np.log2(dims)*8

# DENSE SOFTMAX - SINGLE - AX0
config = {'binary': 'softmax_dense', 'num_proc': 1, 'dimensions': dims, 'num_runs': 1, 'axis': -1}
sc_dense_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
sc_dense_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
sc_dense_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
# CSR SOFTMAX - SINGLE - AX0
config = {'binary': 'softmax_csr', 'num_proc': 1, 'dimensions': dims, 'num_runs': 10, 'axis': -1}
sc_csr0_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
sc_csr0_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
sc_csr0_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
# CSR SOFTMAX - SINGLE - AX1
config = {'binary': 'softmax_csr', 'num_proc': 1, 'dimensions': dims, 'num_runs': 10, 'axis': 0}
sc_csr1_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
sc_csr1_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
sc_csr1_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
# DENSE SOFTMAX - PARALLEL - AX0
config = {'binary': 'softmax_dense', 'num_proc': 8, 'dimensions': dims, 'num_runs': 1, 'axis': -1}
mc_dense_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
mc_dense_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
mc_dense_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
mc_dense_synchov = create_dataframe2(directory, config, 'mean', 'synch_overhead')
# CSR SOFTMAX -PARALLEL - AX0
config = {'binary': 'softmax_csr', 'num_proc': 8, 'dimensions': dims, 'num_runs': 10, 'axis': -1}
mc_csr0_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
mc_csr0_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
mc_csr0_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
mc_csr0_synchov = create_dataframe2(directory, config, 'mean', 'synch_overhead')
# CSR SOFTMAX -PARALLEL - AX1
config = {'binary': 'softmax_csr', 'num_proc': 8, 'dimensions': dims, 'num_runs': 10, 'axis': 0}
mc_csr1_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
mc_csr1_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
mc_csr1_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
mc_csr1_synchov = create_dataframe2(directory, config, 'mean', 'synch_overhead')

#############################################################################
# CSR SOFTMAX - SINGLE - AX1
config = {'binary': 'softmax_csr_version2', 'num_proc': 1, 'dimensions': dims, 'num_runs': 10, 'axis': 0}
sc_csr12_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
sc_csr12_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
sc_csr12_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
# CSR SOFTMAX -PARALLEL - AX1
config = {'binary': 'softmax_csr_version4', 'num_proc': 8, 'dimensions': dims, 'num_runs': 10, 'axis': 0}
mc_csr12_cycles = create_dataframe2(directory, config, 'mean', 'cycles')
mc_csr12_coreipc = create_dataframe2(directory, config, 'mean', 'snitch_occupancy')
mc_csr12_fpssipc = create_dataframe2(directory, config, 'mean', 'fpss_occupancy')
mc_csr12_synchov = create_dataframe2(directory, config, 'mean', 'synch_overhead')

#############################################################################
# PLOT FEATURES

SMALL_SIZE = 12
MEDIUM_SIZE = 16
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
width = 1.5
m, se, h = mean_confidence_interval(sc_dense_cycles, 0.95)
l1 = ax0.bar(dims_normalized-3*width/2, m, width, color=cmap[0], edgecolor='k')

m, se, h = mean_confidence_interval(sc_csr0_cycles, 0.95)
l2 = ax0.bar(dims_normalized-width/2, m, width, color=cmap[1], edgecolor='k')
ax0.errorbar(dims_normalized-width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval(sc_csr1_cycles, 0.95)
l3 = ax0.bar(dims_normalized+width/2, m, width, color=cmap[2], edgecolor='k')
ax0.errorbar(dims_normalized+width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval(sc_csr12_cycles, 0.95)
l4 = ax0.bar(dims_normalized+3*width/2, m, width, color=cmap[3], edgecolor='k')
ax0.errorbar(dims_normalized+3*width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

ax0.legend([l1, l2, l3, l4], ['dense', 'CSR axis-1', 'CSR axis-2 v1', 'CSR axis-2 v2'], loc='upper left', facecolor='white', framealpha=1)
ax0.set_xticks(dims_normalized, dims)
ax0.set_xlabel('Input dimension')
ax0.set_ylabel('Cycles')
ax0.set_title('SOFTMAX Single-core')
ax0.set(ylim=(0, 10**6), yticks=np.arange(0, 10**6, 10**5))
ax0.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)


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
width = 1.5
m, se, h = mean_confidence_interval(mc_dense_cycles, 0.95)
l1 = ax1.bar(dims_normalized-3*width/2, m, width, color=cmap[0], edgecolor='k')

m, se, h = mean_confidence_interval(mc_csr0_cycles, 0.95)
l2 = ax1.bar(dims_normalized-width/2, m, width, color=cmap[1], edgecolor='k')
ax1.errorbar(dims_normalized-width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval(mc_csr1_cycles, 0.95)
l3 = ax1.bar(dims_normalized+width/2, m, width, color=cmap[2], edgecolor='k')
ax1.errorbar(dims_normalized+width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval(mc_csr12_cycles, 0.95)
l4 = ax1.bar(dims_normalized+3*width/2, m, width, color=cmap[3], edgecolor='k')
ax1.errorbar(dims_normalized+3*width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

ax1.legend([l1, l2, l3, l4], ['dense', 'CSR axis-1', 'CSR axis-2 v1', 'CSR axis-2 v2'], loc='upper left', facecolor='white', framealpha=1)
ax1.set_xticks(dims_normalized, dims)
ax1.set_xlabel('Input dimension')
ax1.set_ylabel('Cycles')
ax1.set_title('SOFTMAX 8-cores')
ax1.set(ylim=(0, 1.2*10**5), yticks=np.arange(0, 1.5*10**5, 5*10**4))
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
ax1.grid(True)
plt.tight_layout()

m, se, h = mean_confidence_interval_ratio(mc_dense_cycles, mc_csr0_cycles, 0.95)
print(np.mean(m))
m, se, h = mean_confidence_interval_ratio(mc_dense_cycles, mc_csr12_cycles, 0.95)
print(np.mean(m))
m, se, h = mean_confidence_interval_ratio(sc_dense_cycles, sc_csr0_cycles, 0.95)
print(np.mean(m))
m, se, h = mean_confidence_interval_ratio(sc_dense_cycles, sc_csr12_cycles, 0.95)
print(np.mean(m))

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
width = 1.5
m, se, h = mean_confidence_interval_ratio(sc_dense_cycles, mc_dense_cycles, 0.95)
l1 = ax2.bar(dims_normalized-3*width/2, m, width, color=cmap[0], edgecolor='k')

m, se, h = mean_confidence_interval_ratio(sc_csr0_cycles, mc_csr0_cycles, 0.95)
l2 = ax2.bar(dims_normalized-width/2, m, width, color=cmap[1], edgecolor='k')
ax2.errorbar(dims_normalized-width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval_ratio(sc_csr1_cycles, mc_csr1_cycles, 0.95)
l3 = ax2.bar(dims_normalized+width/2, m, width, color=cmap[2], edgecolor='k')
ax2.errorbar(dims_normalized+width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

m, se, h = mean_confidence_interval_ratio(sc_csr12_cycles, mc_csr12_cycles, 0.95)
l4 = ax2.bar(dims_normalized+3*width/2, m, width, color=cmap[3], edgecolor='k')
ax2.errorbar(dims_normalized+3*width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

ax2.legend([l1, l2, l3, l4], ['dense', 'CSR axis-1', 'CSR axis-2 v1', 'CSR axis-2 v2'], loc='upper left', facecolor='white', framealpha=1)
ax2.set_xticks(dims_normalized, dims)
ax2.set_xlabel('Input dimension')
ax2.set_ylabel('Speed-up')
ax2.set_title('SOFTMAX Speed-UP')
ax2.set(ylim=(0,10), yticks=np.arange(0, 10, 1))
ax2.text(42, 8.2, 'IDEAL Speed-UP', color='r', fontsize=MEDIUM_SIZE)
ax2.axhline(y = 8, color = 'r', linestyle = '--')
ax2.grid(True)
plt.tight_layout()

#############################################################################
# IPC PLOT

fig1 = plt.figure(figsize=(11, 6))
ax0= plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax1= plt.subplot2grid((2, 2), (1, 0), colspan=2)
cmap = mcp.gen_color(cmap="RdBu",n=6)
cmap=cmap[2:4]
patterns = [ "", "/" , ".", "x" ]

ax0.bar(dims_normalized-3*width/2, sc_dense_coreipc.apply(np.mean, axis=0), width,
        color=cmap[0], hatch=patterns[0], edgecolor='k')
ax0.bar(dims_normalized-3*width/2, sc_dense_fpssipc.apply(np.mean, axis=0), width,
        bottom=sc_dense_coreipc.apply(np.mean, axis=0),
        color=cmap[1], hatch=patterns[0], edgecolor='k')

ax0.bar(dims_normalized-width/2, sc_csr0_coreipc.apply(np.mean, axis=0), width,
        yerr=3*sc_csr0_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax0.bar(dims_normalized-width/2, sc_csr0_fpssipc.apply(np.mean, axis=0), width,
        yerr=sc_csr0_fpssipc.apply(np.std, axis=0), bottom=sc_csr0_coreipc.apply(np.mean, axis=0),
        color=cmap[1], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

ax0.bar(dims_normalized+width/2, sc_csr1_coreipc.apply(np.mean, axis=0), width,
        yerr=3*sc_csr1_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax0.bar(dims_normalized+width/2, sc_csr1_fpssipc.apply(np.mean, axis=0), width,
        yerr=3*sc_csr1_fpssipc.apply(np.std, axis=0), bottom=sc_csr1_coreipc.apply(np.mean, axis=0),
        color=cmap[1], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

ax0.bar(dims_normalized+3*width/2, sc_csr12_coreipc.apply(np.mean, axis=0), width,
        yerr=3*sc_csr12_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[3], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax0.bar(dims_normalized+3*width/2, sc_csr12_fpssipc.apply(np.mean, axis=0), width,
        yerr=3*sc_csr12_fpssipc.apply(np.std, axis=0), bottom=sc_csr12_coreipc.apply(np.mean, axis=0),
        color=cmap[1], hatch=patterns[3], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

ax0.legend( [plt.bar([0], [0], color=cmap[0], edgecolor='k'),
             plt.bar([0], [0], color=cmap[1], edgecolor='k'),
             plt.bar([0], [0], color='w', edgecolor='k'),
             plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[1]),
             plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[2]),
             plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[3])],
            ["INT-core IPC", "FP-SS IPC","Dense", "CSR axis-1", "CSR axis-2 v1", "CSR axis-2 v2"],
            loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='white', framealpha=1)
ax0.set_xticks(dims_normalized, dims)
ax0.set(ylim=(0,1.2), yticks=np.arange(0, 1.3, 0.2))
ax0.text(42, 1.05, 'IDEAL IPC', color='r', fontsize=MEDIUM_SIZE)
ax0.axhline(y = 1, color = 'r', linestyle = '--')
ax0.set_xlabel('Input dimension')
ax0.set_ylabel('Occupation')
ax0.set_title('SOFTMAX Single-core')
ax0.grid(True)
plt.tight_layout()


s1 = mc_dense_coreipc.apply(np.mean, axis=0)
s2 = mc_dense_fpssipc.apply(np.mean, axis=0)
s3 = mc_dense_synchov.apply(np.mean, axis=0)
ax1.bar(dims_normalized-3*width/2, s1, width, color=cmap[0], hatch=patterns[0], edgecolor='k')
ax1.bar(dims_normalized-3*width/2, s2, width, bottom=s1, color=cmap[1], hatch=patterns[0], edgecolor='k')
ax1.bar(dims_normalized-3*width/2, s3, width, bottom=s1+s2, color='lightgray', edgecolor='k')

s1 = mc_csr0_coreipc.apply(np.mean, axis=0)
s2 = mc_csr0_fpssipc.apply(np.mean, axis=0)
s3 = mc_csr0_synchov.apply(np.mean, axis=0)
ax1.bar(dims_normalized-width/2, s1, width, yerr=mc_csr0_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized-width/2, s2, width, yerr=mc_csr0_fpssipc.apply(np.std, axis=0), bottom=s1,
        color=cmap[1], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized-width/2, s3, width, yerr=mc_csr0_synchov.apply(np.std, axis=0), bottom=s1+s2,
        color='lightgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

s1 = mc_csr1_coreipc.apply(np.mean, axis=0)
s2 = mc_csr1_fpssipc.apply(np.mean, axis=0)
s3 = mc_csr1_synchov.apply(np.mean, axis=0)
ax1.bar(dims_normalized+width/2, s1, width, yerr=mc_csr1_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized+width/2, s2, width, yerr=mc_csr1_fpssipc.apply(np.std, axis=0), bottom=s1,
        color=cmap[1], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized+width/2, s3, width, yerr=mc_csr1_synchov.apply(np.std, axis=0), bottom=s1+s2,
        color='lightgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

s1 = mc_csr12_coreipc.apply(np.mean, axis=0)
s2 = mc_csr12_fpssipc.apply(np.mean, axis=0)
s3 = mc_csr12_synchov.apply(np.mean, axis=0)
ax1.bar(dims_normalized+3*width/2, s1, width, yerr=mc_csr12_coreipc.apply(np.std, axis=0),
        color=cmap[0], hatch=patterns[3], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized+3*width/2, s2, width, yerr=mc_csr12_fpssipc.apply(np.std, axis=0), bottom=s1,
        color=cmap[1], hatch=patterns[3], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims_normalized+3*width/2, s3, width, yerr=mc_csr12_synchov.apply(np.std, axis=0), bottom=s1+s2,
        color='lightgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

ax1.legend( [plt.bar([3], [0], color=cmap[0], edgecolor='k'),
             plt.bar([3], [0], color=cmap[1], edgecolor='k'),
             plt.bar([3], [0], color='lightgray', edgecolor='k'),
             plt.bar([3], [0], color='w', edgecolor='k'),
             plt.bar([3], [0], color='w', edgecolor='k', hatch=patterns[1]),
             plt.bar([3], [0], color='w', edgecolor='k', hatch=patterns[2]),
             plt.bar([3], [0], color='w', edgecolor='k', hatch=patterns[3])],
            ["INT-core IPC", "FP-SS IPC", "Synch.", "Dense", "CSR axis-1", "CSR axis-2 v1", "CSR axis-2 v2"],
            loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='white', framealpha=1)
ax1.set_xticks(dims_normalized, dims)
ax1.set(ylim=(0,1.2), yticks=np.arange(0, 1.3, 0.2))
ax1.set(xlim=ax0.get_xlim())
ax1.text(42, 1.05, 'IDEAL IPC', color='r', fontsize=MEDIUM_SIZE)
ax1.axhline(y = 1, color = 'r', linestyle = '--')
ax1.set_xlabel('Input dimension')
ax1.set_ylabel('Occupation')
ax1.set_title('SOFTMAX 8-cores')
ax1.grid(True)
plt.tight_layout()


plt.show()





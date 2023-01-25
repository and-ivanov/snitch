#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

import argparse
import pathlib
import glob
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
from matplotlib import font_manager


def read_run_from_csv(indir: pathlib.Path, binary: str, nproc: int, size: int, metric: str = 'cycles', density = None, stat: str = 'mean'):
    data = pd.DataFrame()
    # Read in csv_files
    for s in size:
        if density is not None:
            glob_str = f"{indir}/{binary}_n{nproc}_s{s}_d{density}_r*.csv"
        else:
            glob_str = f"{indir}/{binary}_n{nproc}_s{s}_r*.csv"
        files = glob.glob(glob_str)
        values = []
        for f in files:
            values.append(pd.read_csv(f, index_col=0).loc[stat, metric])
        data[s] = values
    return data


def mean_confidence_interval(data, confidence):
    m = data.apply(np.mean, axis=0)
    se = data.apply(stats.sem)
    n = len(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, se, h


def plot(indir: pathlib.Path, outdir: pathlib.Path):

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
    cmap = mcp.gen_color(cmap="Spectral_r",n=6)

    dims = np.array([8, 16, 32, 64])
    densities = np.array([0.1, 0.2, 0.3, 0.4])
    dims_normalized = np.log2(dims)*8

    """
    Density Plots
    """

    dense_cycles = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=1, size=dims[::-1])
    parallel_dense_cycles = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=8, size=dims[::-1])
    density_speedup = pd.DataFrame()
    parallel_density_speedup = pd.DataFrame()
    for d in densities:
        csr_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_dense', density=d, nproc=1, size=dims[::-1])
        parallel_csr_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', density=d, nproc=8, size=dims[::-1])
        density_speedup[d] = dense_cycles.mean(axis=0)/csr_cycles.mean(axis=0)
        parallel_density_speedup[d] = parallel_dense_cycles.mean(axis=0)/parallel_csr_cycles.mean(axis=0)

    fig4 = plt.figure(figsize=(4, 8))
    ax0 = plt.subplot2grid((2, 1), (0, 0))
    ax1 = plt.subplot2grid((2, 1), (1, 0))
    im = ax0.imshow(density_speedup, cmap="YlGn_r", vmin=0.8, vmax=2.5)
    ax0.set_xticks(np.arange(len(densities)))
    ax0.set_yticks(np.arange(len(dims[::-1])))
    ax0.set_xticklabels(densities)
    ax0.set_yticklabels(dims[::-1])

    for i in range(len(dims[::-1])):
        for j in range(len(densities)):
            color = "black" if density_speedup.iloc[i, j] > 2 else "white"
            ax0.text(j, i, f"{density_speedup.iloc[i, j]:.2f}",
                     ha="center", va="center", color=color, fontsize=MEDIUM_SIZE, fontweight='bold')

    ax0.set_xlabel('Density')
    ax0.set_ylabel('Input dimension')
    ax0.set_title('Single-Core Density Speedup')
    plt.tight_layout()

    im = ax1.imshow(parallel_density_speedup, cmap="YlGn_r", vmin=0.8, vmax=2.5)
    ax1.set_xticks(np.arange(len(densities)))
    ax1.set_yticks(np.arange(len(dims[::-1])))
    ax1.set_xticklabels(densities)
    ax1.set_yticklabels(dims[::-1])

    for i in range(len(dims[::-1])):
        for j in range(len(densities)):
            ax1.text(j, i, f"{parallel_density_speedup.iloc[i, j]:.2f}",
                     ha="center", va="center", color="w", fontsize=MEDIUM_SIZE, fontweight='bold')

    ax1.set_xlabel('Density')
    ax1.set_ylabel('Input dimension')
    ax1.set_title('8-Core Density Speedup')
    plt.tight_layout()

    plt.savefig(outdir / 'gemm_density.png')
    plt.savefig(outdir / 'gemm_density.eps', format='eps')
    plt.savefig(outdir / 'gemm_density.svg', format='svg')

    """
    Cycles Plots
    """
    dense_cycles = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=1, size=dims)
    csr_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_dense', nproc=1, size=dims)
    csr_csr_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_csr', nproc=1, size=dims)
    parallel_dense_cycles = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=8, size=dims)
    parallel_csr_barrier_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims, metric='synch_cycles')
    parallel_csr_transform_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims, metric='transform_cycles')
    parallel_csr_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims)
    parallel_csr_comp_cycles = parallel_csr_cycles-parallel_csr_barrier_cycles-parallel_csr_transform_cycles
    parallel_csr_csr_barrier_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_csr_to_dense', nproc=8, size=dims, metric='synch_cycles')
    parallel_csr_csr_transform_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_csr_to_dense', nproc=8, size=dims, metric='transform_cycles')
    parallel_csr_csr_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_csr_to_dense', nproc=8, size=dims)
    parallel_csr_csr_comp_cycles = parallel_csr_csr_cycles-parallel_csr_csr_barrier_cycles-parallel_csr_csr_transform_cycles
    norm_parallel_dense_cycles = 1/parallel_dense_cycles.div(dense_cycles.iloc[0], axis=1)
    norm_parallel_csr_cycles = 1/parallel_csr_cycles.div(csr_cycles.iloc[0], axis=1)
    norm_parallel_csr_csr_cycles = 1/parallel_csr_csr_cycles.div(csr_csr_cycles.iloc[0], axis=1)

    fig0 = plt.figure(figsize=(15, 6))
    ax0 = plt.subplot2grid((2, 2), (0, 0), rowspan=1)
    ax1 = plt.subplot2grid((2, 2), (1, 0), rowspan=1)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    cmap = mcp.gen_color(cmap="Spectral_r",n=6)

    width = 1.5
    m, se, h = mean_confidence_interval(dense_cycles, 0.95)
    l1 = ax0.bar(dims_normalized-width, m, width, color=cmap[0], edgecolor='k')
    m, se, h = mean_confidence_interval(csr_cycles, 0.95)
    l2 = ax0.bar(dims_normalized, m, width, color=cmap[1], edgecolor='k')
    ax0.errorbar(dims_normalized, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    m, se, h = mean_confidence_interval(csr_csr_cycles, 0.95)
    l3 = ax0.bar(dims_normalized+width, m, width, color=cmap[2], edgecolor='k')
    ax0.errorbar(dims_normalized+width, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

    ax0.set_xticks(dims_normalized, dims)
    ax0.legend([l1, l2, l3], ['dense', 'CSR-dense', 'CSR-CSR'], loc='upper left', facecolor='white', framealpha=1)
    ax0.set_xlabel('Input dimension')
    ax0.set_ylabel('Cycles')
    ax0.set_title('Matmul Single-core')
    ax0.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax0.grid(True)
    plt.tight_layout()

    width = 1.5
    m1, se, h = mean_confidence_interval(parallel_dense_cycles, 0.95)
    l1 = ax1.bar(dims_normalized-width, m1, width, color=cmap[0], edgecolor='k')
    ax1.errorbar(dims_normalized-width, m1, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    m1, se, h = mean_confidence_interval(parallel_csr_cycles, 0.95)
    l2 = ax1.bar(dims_normalized, m1, width, color=cmap[1], edgecolor='k')
    ax1.errorbar(dims_normalized, m1, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    m1, se, h = mean_confidence_interval(parallel_csr_csr_cycles, 0.95)
    l3 = ax1.bar(dims_normalized+width, m1, width, color=cmap[2], edgecolor='k')
    ax1.errorbar(dims_normalized+width, m1, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

    ax1.set_xticks(dims_normalized, dims)
    ax1.legend([l1, l2, l3], ['dense', 'CSR-dense', 'CSR-CSR'], loc='upper left', facecolor='white', framealpha=1)
    ax1.set_xlabel('Input dimension')
    ax1.set_ylabel('Cycles')
    ax1.set_title('Matmul Multi-core')
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
    ax1.grid(True)
    plt.tight_layout()

    # m, se, h = mean_confidence_interval(norm_dense_cycles, 0.95)
    # l1 = ax2.bar(dims_normalized-width, m, width, color=cmap[0], edgecolor='k')
    # ax2.errorbar(dims_normalized-width, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    m, se, h = mean_confidence_interval(norm_parallel_dense_cycles, 0.95)
    l2 = ax2.bar(dims_normalized-width, m, width, color=cmap[0], edgecolor='k')
    ax2.errorbar(dims_normalized-width, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    m, se, h = mean_confidence_interval(norm_parallel_csr_cycles, 0.95)
    l3 = ax2.bar(dims_normalized, m, width, color=cmap[1], edgecolor='k')
    ax2.errorbar(dims_normalized, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    m, se, h = mean_confidence_interval(norm_parallel_csr_csr_cycles, 0.95)
    l4 = ax2.bar(dims_normalized+width, m, width, color=cmap[2], edgecolor='k')
    ax2.errorbar(dims_normalized+width, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    ax2.legend([l2, l3, l4], ['dense', 'CSR-dense', 'CSR-CSR'], loc='upper left', facecolor='white', framealpha=1)
    ax2.set_xticks(dims_normalized, dims)
    ax2.set_xlabel('Input dimension')
    ax2.set_ylabel('Speed-up')
    ax2.set_title('Matmul Speed-UP')
    ax2.set(ylim=(0, 10), yticks=np.arange(0, 10, 1))
    ax2.text(42, 8.2, 'IDEAL Speed-UP', color='r', fontsize=MEDIUM_SIZE)
    ax2.axhline(y=8, color='r', linestyle='--')
    ax2.grid(True)
    plt.tight_layout()

    plt.savefig(outdir / 'gemm_cycles.png')
    plt.savefig(outdir / 'gemm_cycles.eps', format='eps')
    plt.savefig(outdir / 'gemm_cycles.svg', format='svg')

    """
    IPC Plots
    """

    dense_coreipc = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=1, size=dims, metric='snitch_occupancy')
    dense_fpuipc = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=1, size=dims, metric='fpss_occupancy')
    csr_coreipc = read_run_from_csv(indir=indir, binary='matmul_csr_dense', nproc=1, size=dims, metric='snitch_occupancy')
    csr_fpuipc = read_run_from_csv(indir=indir, binary='matmul_csr_dense', nproc=1, size=dims, metric='fpss_occupancy')
    csr_csr_coreipc = read_run_from_csv(indir=indir, binary='matmul_csr_csr', nproc=1, size=dims, metric='snitch_occupancy')
    csr_csr_fpuipc = read_run_from_csv(indir=indir, binary='matmul_csr_csr', nproc=1, size=dims, metric='fpss_occupancy')
    parallel_dense_coreipc = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=8, size=dims, metric='snitch_occupancy')
    parallel_dense_fpuipc = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=8, size=dims, metric='fpss_occupancy')
    parallel_dense_synch = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=8, size=dims, metric='synch_overhead')
    parallel_csr_coreipc = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims, metric='snitch_occupancy')
    parallel_csr_fpuipc = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims, metric='fpss_occupancy')
    parallel_csr_synch = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims, metric='synch_overhead')
    # parallel_csr_transform = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims, metric='transform_overhead')
    parallel_csr_csr_coreipc = read_run_from_csv(indir=indir, binary='matmul_csr_csr_to_dense', nproc=8, size=dims, metric='snitch_occupancy')
    parallel_csr_csr_fpuipc = read_run_from_csv(indir=indir, binary='matmul_csr_csr_to_dense', nproc=8, size=dims, metric='fpss_occupancy')
    parallel_csr_csr_synch = read_run_from_csv(indir=indir, binary='matmul_csr_csr_to_dense', nproc=8, size=dims, metric='synch_overhead')
    # parallel_csr_csr_transform = read_run_from_csv(indir=indir, binary='matmul_csr_csr_to_dense', nproc=8, size=dims, metric='transform_overhead')

    fig1 = plt.figure(figsize=(11, 6))
    ax0 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    cmap = mcp.gen_color(cmap="RdBu", n=6)
    cmap = cmap[2:4]
    patterns = ["", "/", ".", "x"]

    ax0.bar(dims_normalized-width, dense_coreipc.apply(np.mean, axis=0), width,
            color=cmap[0], hatch=patterns[0], edgecolor='k')
    ax0.bar(dims_normalized-width, dense_fpuipc.apply(np.mean, axis=0), width,
            bottom=dense_coreipc.apply(np.mean, axis=0),
            color=cmap[1], hatch=patterns[0], edgecolor='k')
    ax0.bar(dims_normalized, csr_coreipc.apply(np.mean, axis=0), width,
            yerr=3*csr_coreipc.apply(np.std, axis=0),
            color=cmap[0], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax0.bar(dims_normalized, csr_fpuipc.apply(np.mean, axis=0), width,
            yerr=csr_fpuipc.apply(np.std, axis=0), bottom=csr_coreipc.apply(np.mean, axis=0),
            color=cmap[1], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax0.bar(dims_normalized+width, csr_csr_coreipc.apply(np.mean, axis=0), width,
            yerr=3*csr_csr_coreipc.apply(np.std, axis=0),
            color=cmap[0], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax0.bar(dims_normalized+width, csr_csr_fpuipc.apply(np.mean, axis=0), width,
            yerr=3*csr_csr_fpuipc.apply(np.std, axis=0), bottom=csr_csr_coreipc.apply(np.mean, axis=0),
            color=cmap[1], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

    ax0.legend([plt.bar([0], [0], color=cmap[0], edgecolor='k'),
               plt.bar([0], [0], color=cmap[1], edgecolor='k'),
               plt.bar([0], [0], color='w', edgecolor='k'),
               plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[1]),
               plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[2])],
               ["INT-core IPC", "FP-SS IPC", "dense", "CSR-dense", "CSR-CSR"],
               loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='white', framealpha=1)
    ax0.set_xticks(dims_normalized, dims)
    ax0.set(ylim=(0, 1.2), yticks=np.arange(0, 1.3, 0.2))
    ax0.text(42, 1.05, 'IDEAL IPC', color='r', fontsize=MEDIUM_SIZE)
    ax0.axhline(y=1, color='r', linestyle='--')
    ax0.set_xlabel('Input dimension')
    ax0.set_ylabel('Cycles')
    ax0.set_title('Occupation Single-core')
    ax0.grid(True)
    plt.tight_layout()

    s1 = parallel_dense_coreipc.apply(np.mean, axis=0)
    s2 = parallel_dense_fpuipc.apply(np.mean, axis=0)
    s3 = parallel_dense_synch.apply(np.mean, axis=0)
    ax1.bar(dims_normalized-width, s1, width, color=cmap[0], hatch=patterns[0], edgecolor='k')
    ax1.bar(dims_normalized-width, s2, width, bottom=s1, color=cmap[1], hatch=patterns[0], edgecolor='k')
    ax1.bar(dims_normalized-width, s3, width, bottom=s1+s2, color='lightgray', edgecolor='k')

    s1 = parallel_csr_coreipc.apply(np.mean, axis=0)
    s2 = parallel_csr_fpuipc.apply(np.mean, axis=0)
    s3 = parallel_csr_synch.apply(np.mean, axis=0)
    # s4 = parallel_csr_transform.apply(np.max, axis=0)
    ax1.bar(dims_normalized, s1, width, yerr=parallel_csr_coreipc.apply(np.std, axis=0),
            color=cmap[0], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax1.bar(dims_normalized, s2, width, yerr=parallel_csr_fpuipc.apply(np.std, axis=0), bottom=s1,
            color=cmap[1], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax1.bar(dims_normalized, s3, width, yerr=parallel_csr_synch.apply(np.std, axis=0), bottom=s1+s2,
            color='lightgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    # ax1.bar(dims_normalized, s4, width, yerr=parallel_csr_synch.apply(np.std, axis=0), bottom=s1+s2+s3,
    #         color='darkgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

    s1 = parallel_csr_csr_coreipc.apply(np.mean, axis=0)
    s2 = parallel_csr_csr_fpuipc.apply(np.mean, axis=0)
    s3 = parallel_csr_csr_synch.apply(np.mean, axis=0)
    # s4 = parallel_csr_csr_transform.apply(np.max, axis=0)
    ax1.bar(dims_normalized+width, s1, width, yerr=parallel_csr_csr_coreipc.apply(np.std, axis=0),
            color=cmap[0], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax1.bar(dims_normalized+width, s2, width, yerr=parallel_csr_csr_fpuipc.apply(np.std, axis=0), bottom=s1,
            color=cmap[1], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax1.bar(dims_normalized+width, s3, width, yerr=parallel_csr_csr_synch.apply(np.std, axis=0), bottom=s1+s2,
            color='lightgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    # ax1.bar(dims_normalized+width, s4, width, yerr=parallel_csr_csr_synch.apply(np.std, axis=0), bottom=s1+s2+s3,
    #         color='darkgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

    ax1.legend([plt.bar([3], [0], color=cmap[0], edgecolor='k'),
                plt.bar([3], [0], color=cmap[1], edgecolor='k'),
                plt.bar([3], [0], color='lightgray', edgecolor='k'),
                plt.bar([3], [0], color='w', edgecolor='k'),
                plt.bar([3], [0], color='w', edgecolor='k', hatch=patterns[1]),
                plt.bar([3], [0], color='w', edgecolor='k', hatch=patterns[2])],
               ["INT-core IPC", "FP-SS IPC", "Synch.", "dense", "CSR-dense", "CSR-CSR"],
               loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='white', framealpha=1)
    ax1.set_xticks(dims_normalized, dims)
    ax1.set(ylim=(0, 1.2), yticks=np.arange(0, 1.3, 0.2))
    ax1.set(xlim=ax0.get_xlim())
    ax1.text(42, 1.05, 'IDEAL IPC', color='r', fontsize=MEDIUM_SIZE)
    ax1.axhline(y=1, color='r', linestyle='--')
    ax1.set_xlabel('Input dimension')
    ax1.set_ylabel('Cycles')
    ax1.set_title('Occupation 8-cores')
    ax1.grid(True)
    plt.tight_layout()

    plt.savefig(outdir / 'gemm_ipc.png')
    plt.savefig(outdir / 'gemm_ipc.eps', format='eps')
    plt.savefig(outdir / 'gemm_ipc.svg', format='svg')


def main():

    script_path = Path(__file__).parent.absolute()

    indir = script_path / 'results'
    outdir = script_path / 'results'

    plot(indir, outdir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

import numpy as np
import scipy.sparse as sp
import argparse
import pathlib
import torch
import torch.nn as nn
from mako.template import Template

np.set_printoptions(precision=3)


def gen_data_header_file(outdir: pathlib.Path, tpl: pathlib.Path, **kwargs):

    file = outdir / f"data_{kwargs['name']}.h"

    template = Template(filename=str(tpl))
    with file.open('w') as f:
        f.write(template.render(**kwargs))


def gen_rand_csr_matrix(m: int, n: int, density: float) -> sp.csr_matrix:
    return sp.random(m, n, density, format='csr')

def conv2d(ifmap, weights, padding=1, stride=1):
    n = 1
    ci, ih, iw = ifmap.shape
    co, _,fh, fw = weights.shape

    conv2d = nn.Conv2d(ci, co, (fh, fw), padding=0)
    conv2d.weight = nn.Parameter(weights, requires_grad=False)
    conv2d.bias = nn.Parameter(torch.zeros_like(conv2d.bias, dtype=weights.dtype), requires_grad=False)
    ofmap = conv2d(ifmap)

    return ofmap

def main():

    script_path = pathlib.Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser(description='Generate data for kernels')
    parser.add_argument(
        "-c",
        "--channel_size",
        type=int,
        required=False,
        default=8,
        help='Input & Output channel size'
    )
    parser.add_argument(
        "-n",
        "--num_proc",
        type=int,
        required=False,
        default=8,
        help='Number of processors'
    )
    parser.add_argument(
        "-m",
        "--matrix_size",
        type=int,
        required=False,
        default=4,
        help='Input matrix size'
    )
    parser.add_argument(
        "-f",
        "--filter_size",
        type=int,
        required=False,
        default=3,
        help='Filter matrix size'
    )
    parser.add_argument(
        "-d",
        "--density",
        type=float,
        required=False,
        default=0.1,
        help='Input matrix density'
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=pathlib.Path,
        default=script_path,
        required=False,
        help='Select out directory of generated data files'
    )
    parser.add_argument(
        "-t_ccc",
        "--tpl_ccc",
        type=pathlib.Path,
        required=False,
        default=script_path / "data_conv2d_ccc.h.tpl",
        help='Path to mako template'
    )
    parser.add_argument(
        "-t_ccd",
        "--tpl_ccd",
        type=pathlib.Path,
        required=False,
        default=script_path / "data_conv2d_ccd.h.tpl",
        help='Path to mako template'
    )
    parser.add_argument(
        "-t_ddd",
        "--tpl_ddd",
        type=pathlib.Path,
        required=False,
        default=script_path / "data_conv2d_ddd.h.tpl",
        help='Path to mako template'
    )
    parser.add_argument(
        "-t_cdc",
        "--tpl_cdc",
        type=pathlib.Path,
        required=False,
        default=script_path / "data_conv2d_cdc.h.tpl",
        help='Path to mako template'
    )
    parser.add_argument(
        "-t_cdd",
        "--tpl_cdd",
        type=pathlib.Path,
        required=False,
        default=script_path / "data_conv2d_cdd.h.tpl",
        help='Path to mako template'
    )
    parser.add_argument(
        "-t_dcd",
        "--tpl_dcd",
        type=pathlib.Path,
        required=False,
        default=script_path / "data_conv2d_dcd.h.tpl",
        help='Path to mako template'
    )
    parser.add_argument(
        "-t_dcdcsrr",
        "--tpl_dcdcsrr",
        type=pathlib.Path,
        required=False,
        default=script_path / "data_conv2d_dcd_csrr.h.tpl",
        help='Path to mako template'
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help='Set verbose'
    )

    args = parser.parse_args()

    ###############################
    ######## Matrices Gen #########
    ###############################
    channel_size          = args.channel_size
    num_proc              = args.num_proc
    matrix_size           = args.matrix_size
    filter_size           = args.filter_size
    density               = args.density
    A_dense_elements      = matrix_size * matrix_size
    filter_dense_elements = filter_size * filter_size
    RES_dense_elements    = (matrix_size - filter_size + 1) * (matrix_size - filter_size + 1)
    
    A = []
    for i in range(channel_size):
        A.append(gen_rand_csr_matrix(m=matrix_size, n=matrix_size, density=density))

    FILTER = []
    for i in range(channel_size):
        FIL = []
        for j in range(channel_size):
            FIL.append(gen_rand_csr_matrix(m=filter_size, n=filter_size, density=density))
        FILTER.append(FIL)

    ###############################
    ######## Golden Model #########
    ###############################
    A_torch = {}
    A_TENSOR = []
    for i in range(channel_size):
        A_torch[i] = torch.from_numpy(A[i].todense())
        A_TENSOR.append(A_torch[i])
    A_TENSOR = torch.stack(A_TENSOR, 0)

    F_TENSOR = []
    for i in range(channel_size):
        F_torch = {}
        F_TEN = []
        for j in range(channel_size):
            F_torch[j] = torch.from_numpy(FILTER[i][j].todense())
            F_TEN.append(F_torch[j])
        F_TENSOR.append(torch.stack(F_TEN, 0)) 
    F_TENSOR = torch.stack(F_TENSOR,0)

    RES_TENSOR = conv2d(ifmap=A_TENSOR, weights=F_TENSOR, padding=0, stride=1)
    RES_TENSOR = RES_TENSOR.detach()
    RES = []
    for i in range(channel_size):
        RES.append(sp.csr_matrix(RES_TENSOR[i].numpy()))

    ###############################
    ######## Output  File #########
    ###############################
    kwargs = {'name': 'conv2d_csr_csr_csr', 'A' : A, 'FILTER': FILTER, 'RES' : RES, 'channel_size' : channel_size, 'num_proc' : num_proc}
    gen_data_header_file(args.outdir, args.tpl_ccc, **kwargs)
    
    kwargs = {'name': 'conv2d_csr_csr_dense', 'A' : A, 'FILTER': FILTER, 'RES' : RES, 'channel_size' : channel_size, 'num_proc' : num_proc, 'RES_dense_elements' : RES_dense_elements}
    gen_data_header_file(args.outdir, args.tpl_ccd, **kwargs)

    kwargs = {'name': 'conv2d_dense_dense_dense', 'A' : A, 'FILTER': FILTER, 'RES' : RES, 'channel_size' : channel_size, 'num_proc' : num_proc, 'A_dense_elements' : A_dense_elements, 'filter_dense_elements' : filter_dense_elements, 'RES_dense_elements' : RES_dense_elements}
    gen_data_header_file(args.outdir, args.tpl_ddd, **kwargs)

    kwargs = {'name': 'conv2d_csr_dense_csr', 'A' : A, 'FILTER': FILTER, 'RES' : RES, 'channel_size' : channel_size, 'num_proc' : num_proc, 'filter_dense_elements' : filter_dense_elements}
    gen_data_header_file(args.outdir, args.tpl_cdc, **kwargs)
    
    kwargs = {'name': 'conv2d_csr_dense_dense', 'A' : A, 'FILTER': FILTER, 'RES' : RES, 'channel_size' : channel_size, 'num_proc' : num_proc, 'filter_dense_elements' : filter_dense_elements, 'RES_dense_elements' : RES_dense_elements}
    gen_data_header_file(args.outdir, args.tpl_cdd, **kwargs)

    kwargs = {'name': 'conv2d_dense_csr_dense', 'A' : A, 'FILTER': FILTER, 'RES' : RES, 'channel_size' : channel_size, 'num_proc' : num_proc, 'A_dense_elements' : A_dense_elements, 'RES_dense_elements' : RES_dense_elements}
    gen_data_header_file(args.outdir, args.tpl_dcd, **kwargs)

    kwargs = {'name': 'conv2d_dense_csrr_dense', 'A' : A, 'FILTER': FILTER, 'RES' : RES, 'channel_size' : channel_size, 'num_proc' : num_proc, 'A_dense_elements' : A_dense_elements, 'RES_dense_elements' : RES_dense_elements}
    gen_data_header_file(args.outdir, args.tpl_dcdcsrr, **kwargs)

if __name__ == "__main__":
    main()
    
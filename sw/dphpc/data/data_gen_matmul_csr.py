#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

import numpy as np
import scipy.sparse as sp
import argparse
import pathlib
from mako.template import Template

np.set_printoptions(precision=3)


def gen_data_header_file(outdir: pathlib.Path.cwd(), tpl: pathlib.Path.cwd(), **kwargs):

    file = outdir / f"data_{kwargs['name']}.h"

    template = Template(filename=str(tpl))
    with file.open('w') as f:
        f.write(template.render(**kwargs))


def gen_rand_matrix(m: int, n: int, density: float, fmt='csr') -> sp.csr_matrix:
    return sp.random(m, n, density=density, format=fmt)


def main():

    script_path = pathlib.Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser(description='Generate data for kernels')
    parser.add_argument(
        "-o",
        "--outdir",
        type=pathlib.Path,
        default=script_path,
        required=False,
        help='Select out directory of generated data files'
    )
    parser.add_argument(
        "-t",
        "--tpl",
        type=pathlib.Path,
        required=False,
        default=script_path / "data.h.tpl",
        help='Path to mako template'
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help='Set verbose'
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        required=False,
        default=32,
        help='Size of matrices'
    )
    parser.add_argument(
        "-d",
        "--density",
        type=float,
        required=False,
        default=0.1,
        help='Density of matrices'
    )
    parser.add_argument(
        "-n",
        "--nproc",
        type=int,
        required=False,
        default=8,
        help='Number of processing cores'
    )
    parser.add_argument(
        "-m",
        "--measurement",
        action='store_true',
        help='Skip correctness check'
    )

    args = parser.parse_args()

    A = gen_rand_matrix(m=args.size, n=args.size, density=args.density, fmt='csr')
    B = gen_rand_matrix(m=args.size, n=args.size, density=args.density, fmt='csr')
    C = A * B
    C.sort_indices()

    kwargs = {'name': 'matmul', 'A': A, 'B': B, 'C': C, 'nproc': args.nproc, 'measurement': args.measurement}

    gen_data_header_file(args.outdir, args.tpl, **kwargs)


if __name__ == "__main__":
    main()

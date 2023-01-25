# DPHPC Project: Parallel Implementation of Sparse ONNX Kernels on a Cluster of Lightweight Processors

## Introduction

This folder contains all the source code and the documentation for the DPHPC project. The project is a parallel implementation of sparse ONNX kernels on a cluster of lightweight processors. The project is based on the [Snitch](https://github.com/pulp-platform/snitch) project. We implemented sparse kernels for Softmax, GEMM and Convolution operators. All kernels have been verified and profiled with banshee and with cycle-accurate RTL simulations.

The authors of the project are:
* Tim Fischer `<fischeti@iis.ee.ethz.ch>`
* Marco Bertuletti `<mbertuletti@iis.ee.ethz.ch>`
* Yichao Zhang `<yiczhang@iis.ee.ethz.ch>`

## Contents
* `data`: Data generation scripts and template files for all kernels
   * `data_gen_<kernel>.py`: Python scripts which randomly generate data for the kernels and also calculate the expected output with golden models based on Scipy, Tensorflow etc.
   * `data_<kernel>.h.tpl`: Template files for the data generation scripts. The scripts will fill the template files with the generated data.
   * `data_<kernel>.h`: Generated data files for the kernels.
* `src`: Source code of all SW testbenches and kernels
    * `main_<kernel>.c`: Testbenches for the kernels, each testbench loads the data into L1 before calling the kernel. The testbenches also check the output against the expected output. The performance is also measured with performance counters that are set right before and after calling the corresponding kernel.
    * `kernels`: ONNX kernels for Softmax, GEMM and Convolution
    * `utils`: Some utility functions

## Prerequisites

We recommend to follow [this guide](https://pulp-platform.github.io/snitch/ug/getting_started/) to setup Snitch.

Further, `python3` is required with the packages specified in the `requirements.txt`

## Usage

### Basic usage

To run a binary on Banshee, please type the following commands:

```bash
mkdir build && cd build
cmake ..
make run-banshee-<kernel>_<formats>
```

where `<kernel>` is one of `softmax`, `matmul` or `conv` and `<formats>` are the corresponding sparse representations.
For example:

```bash
make run-banshee-matmul_csr_dense
```

To run the kernels on cycle-accurate RTL simulations, you need to compile the simulator first. We recommend to use the openly available verilator simulator. Please refer to [this guide](https://github.com/pulp-platform/snitch/blob/master/hw/system/snitch_cluster/README.md) to compile the verilator executable of a Snitch cluster.

Afterwards, you can run the kernels with the following commands:

```bash
mkdir build && cd build
cmake -DSNITCH_RUNTIME snRuntime-cluster ..
make run-rtl-<kernel>_<formats>
```


where `<kernel>` and `<formats>` behave as described above.

## Measurements

To reproduce our results we wrote a script that automizes RTL measurements and spits out a CSV file with the results. The script is located in `measurements.py`. The script requires the verilator executable of a Snitch cluster, which can be compiled as described above. The script requires a test config JSON file (`test_cfg.json`) of all the kernel runs that are done. The runs can be configured the following:

```json
[
  {
    "binary": "matmul_csr_csr_to_dense",
    "nproc": [8],
    "size": [8, 16, 32, 64]
  },
  {
    "binary": "softmax_csr",
    "nproc": [1],
    "size": [8, 16, 32, 64],
    "axis": 0
  }
]
```

where `binary` is the kernel that is run, `nproc` defines on how many cores to parallelize the kernel, `size` defines the size of the input data, and `axis` defines the axis along which the softmax is applied. The script will run all combinations of the parameters and and an additional script `perf_extr.py` will output a CSV file with the results.

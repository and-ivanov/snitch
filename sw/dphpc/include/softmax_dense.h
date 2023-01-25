// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Marco Bertuletti <mbertuletti@iis.ee.ethz.ch>#pragma once

#pragma once

inline double my_exp(double x);
void softmax_dense_single(int axis, double volatile *A, double volatile *res, int Ncols, int Nrows);
void softmax_dense_parallel(int axis, double volatile *A, double volatile *res, int core_id, int nPE, int Ncol, int Nrow);

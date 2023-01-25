// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Marco Bertuletti <mbertuletti@iis.ee.ethz.ch>#pragma once

#pragma once

#include "matrix_types.h"

double my_fabs(double x);
inline double my_exp(double x);
void softmax_csr_single(int axis, csr_matrix volatile *A, double volatile *res);
void softmax_csr_parallel(int axis, csr_matrix volatile *A, double volatile *res, int core_id, int nPE);




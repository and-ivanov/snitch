// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#pragma once

#include "matrix_types.h"

void matmul_csr_csr(csr_matrix *A, csr_matrix *B, csr_matrix *res);

void matmul_csr_csr_to_dense(csr_matrix *A, csr_matrix *B, dense_matrix *res);

void matmul_csr_dense(csr_matrix *A, dense_matrix *B, csr_matrix *res);

void matmul_dense_dense(dense_matrix *A, dense_matrix *B, dense_matrix *res);

void matmul_csr_dense_to_dense(csr_matrix *A, dense_matrix *B, dense_matrix *res);

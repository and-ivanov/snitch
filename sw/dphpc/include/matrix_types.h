// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

typedef struct csr_matrix {
  double *values;
  int *col_idx;
  int *row_ptr;
  int nnz;
  int rows;
  int cols;
} csr_matrix;

typedef struct csrr_matrix {
  double *values;
  int *col_idx;
  int *row_idx;
  int *row_ptr;
  int nnz;
  int rows;
  int cols;
} csrr_matrix;

typedef struct dense_matrix {
  double *values;
  int rows;
  int cols;
} dense_matrix;

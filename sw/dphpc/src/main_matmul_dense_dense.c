// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#include <math.h>
#include "data_matmul.h"
#include "matmul_csr.h"
#include "utils.h"
#include "snrt.h"
#include "printf.h"

dense_matrix *matrix_A, *matrix_B, *matrix_res;

int main() {

  const int compute_id = snrt_cluster_compute_core_idx();
  const int compute_num = snrt_cluster_compute_core_num();

  if (snrt_is_dm_core()) {
    // Allocate memory for matrices struct
    matrix_A = snrt_l1alloc(sizeof(dense_matrix));
    matrix_B = snrt_l1alloc(sizeof(dense_matrix));
    matrix_res = snrt_l1alloc(sizeof(dense_matrix));

    // Allocate memory for matrix data A
    matrix_A->values = snrt_l1alloc(sizeof(A_data_dense));
    matrix_A->rows = A.rows;
    matrix_A->cols = A.cols;

    // Allocate memory for matrix data B
    matrix_B->values = snrt_l1alloc(sizeof(B_data_dense));
    matrix_B->rows = B.rows;
    matrix_B->cols = B.cols;

    // Allocate memory for matrix data res
    matrix_res->values = snrt_l1alloc(sizeof(C_data_dense));

    // Copy matrix data to L1
    snrt_dma_start_1d((void *)matrix_A->values, (void *)A_data_dense, sizeof(A_data_dense));
    snrt_dma_start_1d((void *)matrix_B->values, (void *)B_data_dense, sizeof(B_data_dense));

    // Wait for DMA to finish
    snrt_dma_wait_all();
  }

  // Wait for all cores to finish DMA
  snrt_cluster_hw_barrier();

  if (snrt_cluster_compute_core_idx() >= NUM_COMP_CORES || snrt_is_dm_core()) {
    snrt_cluster_hw_barrier();
    return 0;
  }

  // Run the matrix multiplication
  if (NUM_COMP_CORES == 1) {
    benchmark_get_cycle();
    matmul_dense_dense(matrix_A, matrix_B, matrix_res);
    benchmark_get_cycle();
  } else {
    benchmark_get_cycle();
    // Create a new matrix struct for each core with
    // a subset of the matrix data
    dense_matrix matrix_A_parallel = *matrix_A;
    dense_matrix matrix_res_parallel = *matrix_res;
    // Divide the rows of the matrix by the number of cores
    matrix_A_parallel.rows = matrix_A->rows / NUM_COMP_CORES;
    // Add an offset to the data pointers
    matrix_A_parallel.values = matrix_A->values + compute_id * matrix_A_parallel.rows * matrix_A->cols;
    matrix_res_parallel.values = matrix_res->values + compute_id * matrix_A_parallel.rows * matrix_B->cols;
    // Run the matrix multiplication with the subset of the data
    matmul_dense_dense(&matrix_A_parallel, matrix_B, &matrix_res_parallel);
    benchmark_get_cycle();
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  benchmark_get_cycle();

  // Check the result
  int errors = 0;
#ifndef MEASUREMENT
  if (compute_id == 0) {
    for (unsigned int i = 0; i < (int)sizeof(C_data_dense)/sizeof(double); i++) {

      if (fabs(matrix_res->values[i] - C_data_dense[i]) > 0.001) {
        errors++;
      }
    }

    if (errors != 0) {
      printf("Errors: %d/%d!\n", errors, sizeof(C_data_dense)/sizeof(double));
    }
  }
#endif
  return errors;
}

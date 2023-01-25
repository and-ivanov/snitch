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

csr_matrix *matrix_A, *matrix_B, *matrix_res;
dense_matrix *matrix_res_dense;

int main() {

  const int compute_id = snrt_cluster_compute_core_idx();
  const int compute_num = snrt_cluster_compute_core_num();

  if (snrt_is_dm_core()) {
    // Allocate memory for matrices struct
    matrix_A = snrt_l1alloc(sizeof(csr_matrix));
    matrix_B = snrt_l1alloc(sizeof(csr_matrix));
    matrix_res = snrt_l1alloc(sizeof(csr_matrix));
    matrix_res_dense = snrt_l1alloc(sizeof(dense_matrix));

    // Allocate memory for matrix data A
    matrix_A->values = snrt_l1alloc(sizeof(A_data));
    matrix_A->col_idx = snrt_l1alloc(sizeof(A_indices));
    matrix_A->row_ptr = snrt_l1alloc(sizeof(A_indptr));
    matrix_A->nnz = A.nnz;
    matrix_A->rows = A.rows;
    matrix_A->cols = A.cols;

    // Allocate memory for matrix data B
    matrix_B->values = snrt_l1alloc(sizeof(B_data));
    matrix_B->col_idx = snrt_l1alloc(sizeof(B_indices));
    matrix_B->row_ptr = snrt_l1alloc(sizeof(B_indptr));
    matrix_B->nnz = B.nnz;
    matrix_B->rows = B.rows;
    matrix_B->cols = B.cols;

    // Allocate memory for dense matrix data res
    matrix_res_dense->values = snrt_l1alloc(sizeof(C_data_dense));
    matrix_res_dense->rows = C.rows;
    matrix_res_dense->cols = C.cols;

    // Allocate memory for matrix data res
    matrix_res->values = snrt_l1alloc(sizeof(C_data));
    matrix_res->col_idx = snrt_l1alloc(sizeof(C_indices));
    matrix_res->row_ptr = snrt_l1alloc(sizeof(C_indptr));

    // Copy matrix data to L1
    snrt_dma_start_1d((void *)matrix_A->values, (void *)A_data, sizeof(A_data));
    snrt_dma_start_1d((void *)matrix_A->col_idx, (void *)A_indices, sizeof(A_indices));
    snrt_dma_start_1d((void *)matrix_A->row_ptr, (void *)A_indptr, sizeof(A_indptr));
    snrt_dma_start_1d((void *)matrix_B->values, (void *)B_data, sizeof(B_data));
    snrt_dma_start_1d((void *)matrix_B->col_idx, (void *)B_indices, sizeof(B_indices));
    snrt_dma_start_1d((void *)matrix_B->row_ptr, (void *)B_indptr, sizeof(B_indptr));


    // Wait for DMA to finish
    snrt_dma_wait_all();
  }

  // Wait for all cores to finish DMA
  snrt_cluster_hw_barrier();

  if (snrt_cluster_compute_core_idx() >= NUM_COMP_CORES || snrt_is_dm_core()) {
    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();
    return 0;
  }

  // Run the matrix multiplication
  if (NUM_COMP_CORES == 1) {
    benchmark_get_cycle();
    matmul_csr_csr_to_dense(matrix_A, matrix_B, matrix_res_dense);
    benchmark_get_cycle();
  } else {
    benchmark_get_cycle();
    // Create a new matrix struct for each core with
    // a subset of the matrix data
    csr_matrix matrix_A_parallel = *matrix_A;
    dense_matrix matrix_res_parallel = *matrix_res_dense;
    // Divide the rows of the matrix by the number of cores
    matrix_A_parallel.rows = matrix_A->rows / NUM_COMP_CORES;
    // Add an offset to the data pointers
    matrix_A_parallel.row_ptr = matrix_A->row_ptr + matrix_A_parallel.rows * compute_id;
    matrix_res_parallel.values = matrix_res_dense->values + compute_id * matrix_A_parallel.rows * matrix_B->cols;
    // Run the matrix multiplication with the subset of the data
    matmul_csr_csr_to_dense(&matrix_A_parallel, matrix_B, &matrix_res_parallel);
    benchmark_get_cycle();
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  benchmark_get_cycle();
  if (compute_id == 0) {
    dense_to_csr(matrix_res_dense, matrix_res);
  }
  snrt_cluster_hw_barrier();
  benchmark_get_cycle();

  // Check the result
  int errors = 0;
#ifndef MEASUREMENT
  for (int i = 0; i < matrix_res->nnz; i++) {

    if (fabs(matrix_res->values[i] - C.values[i]) > 0.001) {
      errors++;
    }
  }

  if (errors != 0) {
    printf("Errors: %d/%d!\n", errors, matrix_res->nnz);
  }
#endif
  return errors;
}

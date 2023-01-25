// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#include "snrt.h"
#include "utils.h"
#include "printf.h"

#include "data_softmax_csr.h"
#include "softmax_csr.h"

csr_matrix volatile matrix_res;
csr_matrix volatile matrix_A;

int volatile errors;
double volatile ERROR = 1e-1;

int main() {

    int volatile core_id = snrt_cluster_core_idx();
    int nPE = snrt_cluster_core_num();

#if (N_PROC == 1)

    if (core_id != 0) return 0;

    // Allocate space for the result matrix
    matrix_res.values = snrt_l1alloc(2 * A.rows * A.cols * sizeof(int));
    matrix_res.col_idx = snrt_l1alloc(A.rows * A.cols * sizeof(int));
    matrix_res.row_ptr = snrt_l1alloc((A.rows + 1) * sizeof(int));
    // Allocate space for the input matrix
    matrix_A.values = snrt_l1alloc(2 * A.nnz * sizeof(int));
    matrix_A.col_idx = snrt_l1alloc(A.nnz * sizeof(int));
    matrix_A.row_ptr = snrt_l1alloc((A.rows + 1) * sizeof(int));

    // Initialize the result matrix
    matrix_res.rows = A.rows;
    matrix_res.cols = A.cols;
    matrix_res.nnz = A.rows * A.cols;
    // Initialize the input matrix
    matrix_A.rows = A.rows;
    matrix_A.cols = A.cols;
    matrix_A.nnz = A.nnz;
    for(int i = 0; i < A.nnz; i++) {
        matrix_A.values[i] = A.values[i];
        matrix_A.col_idx[i] = A.col_idx[i];
    }
    for(int i = 0; i < (A.rows + 1); i++) {
        matrix_A.row_ptr[i] = A.row_ptr[i];
    }
    for(int i = 0; i < A.rows*A.cols; i++) {
        matrix_res.values[i] = 0.0;
    }
    printf("A.values[0] = %f\n", A.values[0]);

    // Run the softmax
    //softmax_csr_single(AXIS, &matrix_A, matrix_res.values);
    size_t time_init = benchmark_get_cycle();
    softmax_csr_single(AXIS, &matrix_A, matrix_res.values);
    size_t time_end = benchmark_get_cycle();

    // Check the result
    for (int i = 0; i < matrix_res.nnz; i++) {
        //printf("matrix_res->values[%d] = %f\n", i, matrix_res.values[i]);
        if (my_fabs(matrix_res.values[i] - C.values[i]) > ERROR) {
            errors++;
        }
    }
    if (errors == 0) {
        printf("Test passed!\n");
    }

    return errors;

#elif (N_PROC == 8)

    if (core_id == 0) {
        // Allocate space for the result matrix
        matrix_res.values = snrt_l1alloc(2 * A.rows * A.cols * sizeof(int));
        matrix_res.col_idx = snrt_l1alloc(A.rows * A.cols * sizeof(int));
        matrix_res.row_ptr = snrt_l1alloc((A.rows + 1) * sizeof(int));
        // Allocate space for the input matrix
        matrix_A.values = snrt_l1alloc(2 * A.nnz * sizeof(int));
        matrix_A.col_idx = snrt_l1alloc(A.nnz * sizeof(int));
        matrix_A.row_ptr = snrt_l1alloc((A.rows + 1) * sizeof(int));

        // Initialize the result matrix
        matrix_res.rows = A.rows;
        matrix_res.cols = A.cols;
        matrix_res.nnz = A.rows * A.cols;
        // Initialize the input matrix
        matrix_A.rows = A.rows;
        matrix_A.cols = A.cols;
        matrix_A.nnz = A.nnz;
        for(int i = 0; i < A.nnz; i++) {
            matrix_A.values[i] = A.values[i];
            matrix_A.col_idx[i] = A.col_idx[i];
        }
        for(int i = 0; i < (A.rows + 1); i++) {
            matrix_A.row_ptr[i] = A.row_ptr[i];
        }
        printf("A.values[0] = %f\n", A.values[0]);
    }
    snrt_cluster_hw_barrier();

#if (VERSION == 1)

    benchmark_get_cycle();
    if (core_id < 8)
        softmax_csr_parallel(AXIS, &matrix_A, matrix_res.values, core_id, nPE-1);
    benchmark_get_cycle();
    snrt_cluster_hw_barrier();
    benchmark_get_cycle();

#else
    benchmark_get_cycle();
    softmax_csr_parallel(AXIS, &matrix_A, matrix_res.values, core_id, nPE-1);
    benchmark_get_cycle();
    snrt_cluster_hw_barrier();
    benchmark_get_cycle();
#endif

    if (core_id != 0) return 0;
    // Check the result
    errors = 0;
    for (int i = 0; i < A.rows * A.cols; i++) {
        // printf("matrix_res.values[%d] = %f\n", i, matrix_res.values[i]);
        if (my_fabs(matrix_res.values[i] - C.values[i]) > ERROR) {
            errors++;
        }
    }
    if (errors == 0) {
        printf("Test passed!\n");
    }

    return errors;

#endif

}

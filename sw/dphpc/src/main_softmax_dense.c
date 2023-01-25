// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#include "snrt.h"
#include "utils.h"
#include "printf.h"

#include "data_softmax_dense.h"
#include "softmax_dense.h"
#include "softmax_csr.h"

double volatile *matrix_res;
double volatile *matrix_A;

int volatile errors;
double volatile ERROR = 1e-2;

int main() {

    int volatile core_id = snrt_cluster_core_idx();
    int nPE = snrt_cluster_core_num();

#if (N_PROC == 1)

    if (core_id != 0) return 0;

    // Allocate space for the result matrix
    matrix_res = snrt_l1alloc(2 * N * N * sizeof(int));
    // Allocate space for the input matrix
    matrix_A = snrt_l1alloc(2 * N * N * sizeof(int));
    for(int i = 0; i < N * N; i++) {
        matrix_A[i] = A[i];
    }
    printf("A[0] = %f\n", A[0]);

    // Run the softmax
    softmax_dense_single(AXIS, matrix_A, matrix_res, N, N);
    size_t time_init = benchmark_get_cycle();
    softmax_dense_single(AXIS, matrix_A, matrix_res, N, N);
    size_t time_end = benchmark_get_cycle();

    // Check the result
    for (int i = 0; i < N * N; i++) {
        if (my_fabs(matrix_res[i] - C[i]) > ERROR) {
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
        matrix_res = snrt_l1alloc(2 * N * N * sizeof(int));
        // Allocate space for the input matrix
        matrix_A = snrt_l1alloc(2 * N * N * sizeof(int));
        for(int i = 0; i < N * N; i++) {
            matrix_A[i] = A[i];
        }
        printf("A[0] = %f\n", A[0]);
    }
    snrt_cluster_hw_barrier();

    benchmark_get_cycle();
    if (core_id < 8)
        softmax_dense_parallel(AXIS, matrix_A, matrix_res, core_id, nPE - 1, N, N);
    benchmark_get_cycle();
    snrt_cluster_hw_barrier();
    benchmark_get_cycle();

    if (core_id != 0) return 0;
    // Check the result
    for (int i = 0; i < N * N; i++) {
        // printf("matrix_res[%d] = %f\n", i, matrix_res[i]);
        if (my_fabs(matrix_res[i] - C[i]) > ERROR) {
            errors++;
        }
    }
    if (errors == 0) {
        printf("Test passed!\n");
    }
    return errors;

#endif

}

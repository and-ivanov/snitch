// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "utils.h"

#include "../../vendor/riscv-opcodes/encoding.h"

uint32_t benchmark_get_cycle() { return read_csr(mcycle); }

/**
 * @brief fast memset function performed by DMA
 *
 * @param ptr pointer to the start of the region
 * @param value value to set
 * @param len number of bytes, must be multiple of DMA bus-width
 */
void dma_memset(void *ptr, uint8_t value, uint32_t len) {
    // set first 64bytes to value
    // memset(ptr, value, 64);
    uint8_t *p = ptr;
    uint32_t nbytes = 64;
    while (nbytes--) {
        *p++ = value;
    }

    // DMA copy the the rest
    snrt_dma_txid_t memset_txid =
        snrt_dma_start_2d(ptr, ptr, 64, 64, 0, len / 64);
    snrt_dma_wait_all();
}

void dense_to_csr(dense_matrix *A, csr_matrix *res) {

    res->nnz = 0;
    res->rows = A->rows;
    res->cols = A->cols;

    for (uint32_t i = 0; i < A->rows; i++) {
        res->row_ptr[i] = res->nnz;
        for (uint32_t j = 0; j < A->cols; j++) {
            if (A->values[i * A->cols + j] != 0) {
                res->col_idx[res->nnz] = j;
                res->values[res->nnz] = A->values[i * A->cols + j];
                res->nnz++;
            }
        }
    }
    res->row_ptr[A->rows] = res->nnz;
}

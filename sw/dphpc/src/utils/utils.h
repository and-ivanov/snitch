// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snrt.h"
#include "matrix_types.h"

/**
 * @brief returns cycle number and injects marker
 * to track performance
 *
 * @return uint32_t
 */
uint32_t benchmark_get_cycle();

/**
 * @brief fast memset function performed by DMA
 *
 * @param ptr pointer to the start of the region
 * @param value value to set
 * @param len number of bytes, must be multiple of DMA bus-width
 */
void dma_memset(void* ptr, uint8_t value, uint32_t len);

/**
 * @brief converts a dense matrix to a CSR matrix
 *
 * @param A dense matrix
 * @param res CSR matrix
 */
void dense_to_csr(dense_matrix *A, csr_matrix *res);

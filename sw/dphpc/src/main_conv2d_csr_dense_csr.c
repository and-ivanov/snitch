// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yichao Zhang <yiczhang@iis.ee.ethz.ch>

///////////////////////////////////////////////////////////////////////
//////////////////////////      HEAD        ///////////////////////////
///////////////////////////////////////////////////////////////////////
#include <math.h>
#include "conv2d_csr.h"
#include "utils.h"
#include "snrt.h"
#include "printf.h"
#include "data_conv2d_csr_dense_csr.h"

///////////////////////////////////////////////////////////////////////
//////////////////////////     CONFIG       ///////////////////////////
///////////////////////////////////////////////////////////////////////
// 'NUM_COMP_CORES' = 1: 
//  -> kernel will do serial computation;
// 'CHANNELS' < NUM_COMP_CORES': 
//  -> kernel will do partial parallel computation;
// 'CHANNELS' >= NUM_COMP_CORES': 
//  -> kernel will do full cluster parallel computation;
#define NUM_COMP_CORES 8

// Declare output matrix
csr_matrix *matrix_A[CHANNELS], *matrix_res[CHANNELS];  
dense_matrix *matrix_FILTER[CHANNELS][CHANNELS]; 

///////////////////////////////////////////////////////////////////////
//////////////////////////      MAIN        ///////////////////////////
///////////////////////////////////////////////////////////////////////
int main() {
  // ``````````````````````````//
  //####### Matrix Init #######//
  // ......................... //
  assign_A();
  assign_FILTER();
  assign_RES();
  snrt_cluster_hw_barrier();

  // ``````````````````````````//
  //####### Const Value #######//
  // ......................... //
  const int compute_id = snrt_cluster_compute_core_idx();
  const int filter_row = FILTER_dense[0][0].rows;
  const int filter_col = FILTER_dense[0][0].cols;
  const int res_row    = A[0].rows - FILTER_dense[0][0].rows + 1;
  const int res_col    = A[0].cols - FILTER_dense[0][0].cols + 1;
  
  // ``````````````````````````//
  //####### Matrix Alloc ######//
  // ......................... //
  if (snrt_is_dm_core()) {
    for (int j = 0; j < CHANNELS; j++) {

      // Allocate memory for matrix data A
      matrix_A[j] = snrt_l1alloc(sizeof(csr_matrix));
      matrix_A[j]->values = snrt_l1alloc(A[j].nnz * sizeof(double));
      matrix_A[j]->col_idx = snrt_l1alloc(A[j].nnz  * sizeof(int));
      matrix_A[j]->row_ptr = snrt_l1alloc((A[j].rows+1) * sizeof(int));
      matrix_A[j]->nnz = A[j].nnz;
      matrix_A[j]->rows = A[j].rows;
      matrix_A[j]->cols = A[j].cols;

      // Allocate memory for filter data FILTER
      for (int k = 0; k < CHANNELS; k++) {
        matrix_FILTER[k][j] = snrt_l1alloc(sizeof(dense_matrix));
        matrix_FILTER[k][j]->values = snrt_l1alloc(FILTER_dense[k][j].rows * FILTER_dense[k][j].cols * sizeof(double));
        matrix_FILTER[k][j]->rows = FILTER_dense[k][j].rows;
        matrix_FILTER[k][j]->cols = FILTER_dense[k][j].cols;
      }

      // Allocate memory for matrix data res
      matrix_res[j] = snrt_l1alloc(sizeof(csr_matrix));
      matrix_res[j]->values = snrt_l1alloc(RES[j].nnz * sizeof(double));
      matrix_res[j]->col_idx = snrt_l1alloc(RES[j].nnz  * sizeof(int));
      matrix_res[j]->row_ptr = snrt_l1alloc((RES[j].rows+1) * sizeof(int));

      // Copy matrix data to L1
      snrt_dma_start_1d((void *)matrix_A[j]->values, (void *)A[j].values, A[j].nnz * sizeof(double));
      snrt_dma_start_1d((void *)matrix_A[j]->col_idx, A[j].col_idx, A[j].nnz  * sizeof(int));
      snrt_dma_start_1d((void *)matrix_A[j]->row_ptr, (void *)A[j].row_ptr, (A[j].rows+1) * sizeof(int));
      for (int k = 0; k < CHANNELS; k++) {
        snrt_dma_start_1d((void *)matrix_FILTER[k][j]->values, (void *)FILTER_dense[k][j].values, FILTER_dense[k][j].rows * FILTER_dense[k][j].cols * sizeof(double));
      }
    }
    
    // Wait for DMA to finish
    snrt_dma_wait_all();
  }
  
  // Wait for all cores to finish DMA
  snrt_cluster_hw_barrier();

  // ``````````````````````````//
  //####### Calcul Start ######//
  // ......................... //
  int errors = 0;
  
  // Serial
  #if (NUM_COMP_CORES == 1)
    if (compute_id == 0) {  
      printf("Start Single Core Kernel Calculation \n");
      benchmark_get_cycle();
      for (int i = 0; i < CHANNELS; i++) { 
        conv2d_csr_dense_csr(matrix_A, matrix_FILTER[i], matrix_res[i], CHANNELS, filter_row, filter_col, res_row, res_col);
      }
      benchmark_get_cycle();
    }
  
  // Parallel
  #else
    int num_paral_cores = NUM_COMP_CORES;
    int chnl_core = CHANNELS / NUM_COMP_CORES;    
    if (CHANNELS < NUM_COMP_CORES) {
      num_paral_cores = CHANNELS;
      chnl_core = 1;
    }
    if (compute_id < num_paral_cores) {
      benchmark_get_cycle();
      for (int i= compute_id * chnl_core; i < (compute_id + 1) * chnl_core; i++) {
        conv2d_csr_dense_csr(matrix_A, matrix_FILTER[i], matrix_res[i], CHANNELS, filter_row, filter_col, res_row, res_col);
      }
      benchmark_get_cycle();
    }
  #endif
  
  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  
  // Verification
  if (compute_id == 0) {
    printf("Start Results Verification\n");
    for (int i = 0; i < CHANNELS; i++) {
      for (int j = 0; j < matrix_res[i]->nnz; j++) {
        if (fabs(matrix_res[i]->values[j] - RES[i].values[j]) > 0.001) {
          errors++;
        }
      }
      if (errors != 0) {
        printf("Errors: %d/%d!\n", errors, matrix_res[i]->nnz * CHANNELS);
      }
    }
    if (errors == 0) {
      printf("Congratulation! The Results are Correct!\n");
    }
  }
  
  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  
  return errors;
}

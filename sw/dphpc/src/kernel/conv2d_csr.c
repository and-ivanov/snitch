// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yichao Zhang <yiczhang@iis.ee.ethz.ch>

#include "conv2d_csr.h"
#include "printf.h"

void conv2d_csr_csr_csr(struct csr_matrix **A, struct csr_matrix **filter, struct csr_matrix *res, int channel_in, int filter_row, int res_row, int res_col) {
  double sum;
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j++) {
      sum = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        // CSR Version Inner Loop:
        for (int kx = 0; kx < filter_row; kx ++) {
          for (int kx_f = filter[ci]->row_ptr[kx]; kx_f < filter[ci]->row_ptr[kx + 1]; kx_f++) {
            for (int kx_a = A[ci]->row_ptr[kx + i]; kx_a < A[ci]->row_ptr[kx + i + 1]; kx_a++)  {
              if (filter[ci]->col_idx[kx_f] + j == A[ci]->col_idx[kx_a]) {
                sum += A[ci]->values[kx_a] * filter[ci]->values[kx_f];
                break;
              }
            }
          }
        }
      }
      if (sum != 0.0) {
        res->values[res->nnz] = sum;
        res->col_idx[res->nnz] = j;
        res->nnz++;
      }
    }
    res->row_ptr[i + 1] = res->nnz;
  }
}

void conv2d_csr_csr_dense(struct csr_matrix **A, struct csr_matrix **filter, struct dense_matrix *res, int channel_in, int filter_row, int res_row, int res_col) {
  double sum, sum2;
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j+=2) {
      sum = 0;
      sum2 = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        // CSR Version Inner Loop:
        for (int kx = 0; kx < filter_row; kx ++) {
          for (int kx_f = filter[ci]->row_ptr[kx]; kx_f < filter[ci]->row_ptr[kx + 1]; kx_f++) {
            for (int kx_a = A[ci]->row_ptr[kx + i]; kx_a < A[ci]->row_ptr[kx + i + 1]; kx_a++)  {
              // Unloop #1
              if (filter[ci]->col_idx[kx_f] + j == A[ci]->col_idx[kx_a]) {
                sum += A[ci]->values[kx_a] * filter[ci]->values[kx_f];
              }
              // Unloop #2
              if (filter[ci]->col_idx[kx_f] + j + 1 == A[ci]->col_idx[kx_a]) {
                sum2 += A[ci]->values[kx_a] * filter[ci]->values[kx_f];
                break;
              }
            }
          }
        }
      }
      res->values[i * res_col + j] = sum;
      res->values[i * res_col + j + 1] = sum2;
    }
  }
}

void conv2d_csr_dense_csr(struct csr_matrix **A, struct dense_matrix **filter, struct csr_matrix *res, int channel_in, int filter_row, int filter_col, int res_row, int res_col) {
  int A_col_idx, f_idx;
  double sum;
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j++) {
      sum = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        // CSR Version Inner Loop:
        for (int kx = 0; kx < filter_row; kx ++) {
          for (int kx_a = A[ci]->row_ptr[kx + i]; kx_a < A[ci]->row_ptr[kx + i + 1]; kx_a++)  {
            A_col_idx = A[ci]->col_idx[kx_a];
            // Do MAC if A_col_idx inside the filter scope
            if (A_col_idx >= j) {
              if (A_col_idx <= filter_col + j -1){
                f_idx = kx * filter_col + A_col_idx - j;
                sum += A[ci]->values[kx_a] * filter[ci]->values[f_idx];
              }
            }
          }
        }
      }
      if (sum != 0.0) {
        res->values[res->nnz] = sum;
        res->col_idx[res->nnz] = j;
        res->nnz++;
      }
    }
    res->row_ptr[i + 1] = res->nnz;
  }
}

void conv2d_csr_dense_dense(struct csr_matrix **A, struct dense_matrix **filter, struct dense_matrix *res, int channel_in, int filter_row, int filter_col, int res_row, int res_col) {
  int A_col_idx, f_idx;
  double sum, sum2;
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j+=2) {
      sum = 0;
      sum2 = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        // CSR Version Inner Loop:
        for (int kx = 0; kx < filter_row; kx ++) {
          for (int kx_a = A[ci]->row_ptr[kx + i]; kx_a < A[ci]->row_ptr[kx + i + 1]; kx_a++)  {
            A_col_idx = A[ci]->col_idx[kx_a];
            f_idx = kx * filter_col + A_col_idx - j;
            // Do MAC if A_col_idx inside the filter scope
            // Unloop #1
            if (A_col_idx >= j) {
              if (A_col_idx <= filter_col + j - 1){
                sum += A[ci]->values[kx_a] * filter[ci]->values[f_idx];
              }
              // Unloop #2
              if (A_col_idx >= j + 1) {
                if (A_col_idx <= filter_col + j){
                  sum2 += A[ci]->values[kx_a] * filter[ci]->values[f_idx - 1];
                }
              }
            }
          }
        }
      }
      res->values[i * res_col + j] = sum;
      res->values[i * res_col + j + 1] = sum2;
    }
  }
}

void conv2d_dense_csr_csr(struct dense_matrix **A, struct csr_matrix **filter, struct csr_matrix *res, int channel_in, int filter_row, int A_col, int res_row, int res_col) {
  int A_col_idx, A_row_idx;
  double sum;
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j++) {
      sum = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        // CSR Version Inner Loop:
        for (int kx = 0; kx < filter_row; kx ++) {
          for (int kx_f = filter[ci]->row_ptr[kx]; kx_f < filter[ci]->row_ptr[kx + 1]; kx_f++)  {
            A_col_idx = filter[ci]->col_idx[kx_f] + j;
            A_row_idx = kx + i;
            sum += A[ci]->values[A_row_idx * A_col + A_col_idx] * filter[ci]->values[kx_f];
          }
        }
      } 
      if (sum != 0.0) {
        res->values[res->nnz] = sum;
        res->col_idx[res->nnz] = j;
        res->nnz++;
      }
    }
    res->row_ptr[i + 1] = res->nnz;
  }
}

void conv2d_dense_csr_dense(struct dense_matrix **A, struct csr_matrix **filter, struct dense_matrix *res, int channel_in, int filter_row, int A_col, int res_row, int res_col) {
  int A_col_idx, A_row_idx;
  double sum, sum2;
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j+=2) {
      sum = 0;
      sum2 =0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        // CSR Version Inner Loop:
        for (int kx = 0; kx < filter_row; kx ++) {
          for (int kx_f = filter[ci]->row_ptr[kx]; kx_f < filter[ci]->row_ptr[kx + 1]; kx_f++)  {
            A_col_idx = filter[ci]->col_idx[kx_f] + j;
            A_row_idx = kx + i;
            sum += A[ci]->values[A_row_idx * A_col + A_col_idx] * filter[ci]->values[kx_f];
            sum2 += A[ci]->values[A_row_idx * A_col + A_col_idx + 1] * filter[ci]->values[kx_f];
          }
        }
      } 
      res->values[i * res_col + j] = sum;
      res->values[i * res_col + j + 1] = sum2;
    }
  }
}

void conv2d_dense_csrr_dense(struct dense_matrix **A, struct csrr_matrix **filter, struct dense_matrix *res, int channel_in, int A_col, int res_row, int res_col) {
  int A_col_idx, A_row_idx;
  double sum, sum2;
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j+=2) {
      sum = 0;
      sum2 =0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        //int nnz_total = filter[ci]->nnz + filter[ci+1]->nnz + filter[ci+2]->nnz + filter[ci+3]->nnz;
        //double f_collect[35] = {*filter[ci]->values, *filter[ci+1]->values, *filter[ci+2]->values, *filter[ci+3]->values};
        //int f_col_idx_collect[35] = {*filter[ci]->col_idx, *filter[ci+1]->col_idx, *filter[ci+2]->col_idx, *filter[ci+3]->col_idx};
        //int f_row_idx_collect[35] = {*filter[ci]->col_idx, *filter[ci+1]->col_idx, *filter[ci+2]->col_idx, *filter[ci+3]->col_idx};

        for (int k=0; k < filter[ci]->nnz; k++) {
          A_col_idx = filter[ci]->col_idx[k] + j;
          A_row_idx = filter[ci]->row_idx[k] + i;
          sum += A[ci]->values[A_row_idx * A_col + A_col_idx] * filter[ci]->values[k];
          sum2 += A[ci]->values[A_row_idx * A_col + A_col_idx + 1] * filter[ci]->values[k];
        }
      } 
      if (sum != 0.0) {
        res->values[i * res_col + j] = sum;
        res->values[i * res_col + j + 1] = sum2;
      }
    }
  }
}

void conv2d_dense_dense_dense(struct dense_matrix **A, struct dense_matrix **filter, struct dense_matrix *res, int channel_in, int A_col, int filter_col, int filter_row, int res_row, int res_col) {
  int A_idx, f_idx;
  double sum, sum2;
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j+=2) {
      sum = 0;
      sum2 = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        for (int kx = 0; kx < filter_row; kx ++) {
          for (int ky = 0; ky < filter_col; ky +=3) {
            A_idx = (i + kx) * A_col + j + ky;
            f_idx = kx * filter_col + ky;

            double A0= A[ci]->values[A_idx];
            double F0= filter[ci]->values[f_idx];
            sum  += A0 * F0;
            double A1= A[ci]->values[A_idx + 1];
            double F1= filter[ci]->values[f_idx + 1];
            sum2 += A1 * F0;
            sum  += A1 * F1;
            double A2= A[ci]->values[A_idx + 2];
            double F2= filter[ci]->values[f_idx + 2];
            sum2 += A2 * F1;
            double A3= A[ci]->values[A_idx + 3];
            sum  += A2 * F2;
            sum2 += A3 * F2;
          }
        }
      }
      res->values[i * res_col + j] = sum;
      res->values[i * res_col + j + 1] = sum2;

    }
  }
}

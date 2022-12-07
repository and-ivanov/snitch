// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#include "softmax_csr.h"
#include "snrt.h"
#include "printf.h"

double my_fabs(double x) {
    if(x < 0) {
        return -x;
    } else {
        return x;
    }
}

// INFO: This is a custom function to determine the expponential of a floating point number.
//       We assume here the sum representation of an exponential: exp_n(x) = sum_{i=0}^n (x^i/i!).
//       If two partial sums differ less than epsilon, we can stop the summing.
inline double my_exp(double x) 
{ 
//    const double epsilon = 1e-3;
//    double sum = 0.0;
//    int n = 0;
//    double factorial = 1;
//    double power = 1.0;
//    double term;
//    do {
//        term = power * factorial;
//        sum += term;
//        n += 1;
//        power *= x;
//        factorial /= n;
//    } while (my_fabs(term) >= epsilon);

//    double volatile sum = 1.0 + x;
//    double volatile factorial_2 = 0.5000000000;
//    double volatile factorial_3 = 0.1666666667;
//    double volatile factorial_4 = 0.0416666667;
//    double volatile factorial_5 = 0.0083333333;
//    double volatile factorial_6 = 0.0013888889;
//    double power_2 = x * x;
//    double power_3 = x * power_2;
//    double power_4 = x * power_3;
//    double power_5 = x * power_4;
//    double power_6 = x * power_5;
//    sum += power_2 * factorial_2;
//    sum += power_3 * factorial_3;
//    sum += power_4 * factorial_4;
//    sum += power_5 * factorial_5;
//    sum += power_6 * factorial_6;

    double volatile sum = 1.0 + x;
    double power_2, power_3, power_4, power_5, power_6;
    asm volatile(
        "fmul.d %[power_2], %[x], %[x];"
        : [power_2] "+&f" (power_2)
        : [x] "f" (x)
        :
    );
    double volatile factorial_2 = 0.5000000000;
    double volatile factorial_3 = 0.1666666667;
    double volatile factorial_4 = 0.0416666667;
    double volatile factorial_5 = 0.0083333333;
    double volatile factorial_6 = 0.0013888889;
    asm volatile(
        "fmul.d %[power_3], %[power_2], %[x];"
        "fmul.d %[power_4], %[power_2], %[power_2];"
        "fmul.d %[power_2], %[power_2], %[factorial_2];"
        "fmul.d %[power_5], %[power_3], %[power_2];"
        "fmul.d %[power_6], %[power_3], %[power_3];"
        "fmul.d %[power_3], %[power_3], %[factorial_3];"
        "fmul.d %[power_4], %[power_4], %[factorial_4];"
        "fmul.d %[power_5], %[power_5], %[factorial_5];"
        "fmul.d %[power_6], %[power_6], %[factorial_6];"
        "fadd.d %[sum], %[power_2], %[sum];"
        "fadd.d %[power_3], %[power_3], %[power_4];"
        "fadd.d %[power_5], %[power_5], %[power_6];"
        "fadd.d %[sum], %[sum], %[power_3];"
        "fadd.d %[sum], %[sum], %[power_5];"
        : [power_2] "+&f" (power_2), [power_3] "+&f" (power_3),
          [power_4] "+&f" (power_4), [power_5] "+&f" (power_5), [power_6] "+&f" (power_6), [sum] "+&f" (sum)
        : [factorial_2] "f" (factorial_2), [factorial_3] "f" (factorial_3),
          [factorial_4] "f" (factorial_4), [factorial_5] "f" (factorial_5), [factorial_6] "f" (factorial_6),
          [x] "f" (x)
        :
    );
    return sum; 
}

void softmax_csr_single(int axis, csr_matrix volatile *A, double volatile *res) {

    int i, j, k;
    int n_rows, n_cols, n_nnz;
    int *A_row_ptr, *A_col_idx;
    double *A_values;

    n_rows = A->rows;
    n_cols = A->cols;
    n_nnz = A->nnz;

    A_row_ptr = A->row_ptr;
    A_col_idx = A->col_idx;
    A_values = A->values;

    int elr, elr_next, idx;
    double register logit, sum, max, emax;

    // For axis zero
    if (axis == 0) {

        k = 0;
        for (i = 0; i < n_rows; i++) {
            elr = A_row_ptr[i];
            elr_next = A_row_ptr[i + 1];
            // Compute the sum
            max = 0;
            for (j = elr; j < elr_next; j++) {
                max = max < A_values[j] ? A_values[j] : max;
            }
            // Compute the sum
            sum = (n_cols - (elr_next - elr)) * my_exp(0.0-max);
            for (j = elr; j < elr_next; j++) {
                sum += my_exp(A_values[j]-max);
            }
            // Compute the Logits
            emax = my_exp(0.0-max);
            logit = (1.0 / sum) * emax;
            for (j = 0; j < 4 * (n_cols >> 2U); j += 4) {
                res[i * n_cols + j] = logit;
                res[i * n_cols + j + 1] = logit;
                res[i * n_cols + j + 2] = logit;
                res[i * n_cols + j + 3] = logit;
            }
            while (j < n_cols) {
                res[i * n_cols + j] = logit;
                j++;
            }
            while (k < elr_next) {
                idx = A_col_idx[k];
                res[i * n_cols + idx] *= my_exp(A_values[k]);
                k++;
            }
        }

    // For other axes
    } else {

        k = 0;
        while (k < nnz) {
            idx = A_col_idx[k];
            max = res[idx];
            max = max < A_values[k] ? A_values[k] : max;
            res[idx] = max;
            k++;
        }
        k = 0;
        while (k < nnz) {
            idx = A_col_idx[k];
            sum = res[n_cols + idx];
            max = res[idx];
            sum += my_exp(A_values[k] - max);
            res[n_cols + idx] = sum;
            k++;
        }
        for (j = 0; j < n_cols; j++) {
            sum = res[n_cols + j];
            max = res[j];
            emax = my_exp(0.0-max);
            logit = (1.0 / sum) * emax;
            for (i = 0; i < n_rows; i++) {
                res[i * n_cols + j] = logit;
            }
        }
        k = 0;
        for (i = 0; i < n_rows; i++) {
            elr = A_row_ptr[i];
            elr_next = A_row_ptr[i + 1];
            while (k < elr_next) {
                idx = A_col_idx[k];
                res[i * n_cols + idx] *= my_exp(A_values[k]);
                k++;
            }
        }

//        for (j = 0; j < n_cols; j++) {
//            // Compute the sum
//            sum = n_cols;
//            k = 0;
//            while (k < n_nnz) {
//                if (A_col_idx[k] == j) {
//                    sum -= 1;
//                    sum += my_exp(A_values[k]);
//                }
//                k++;
//            }
//            // Compute the logits
//            logit = (double) (1.0 / sum);
//            k = A_row_ptr[0];
//            for (i = 0; i < n_rows; i++) {
//                elr_next = A_row_ptr[i + 1];
//                res[i * n_cols + j] = logit;
//                while(k < elr_next) {
//                    if (A_col_idx[k] == j) {
//                        res[i * n_cols + j] = logit * my_exp(A_values[k]);
//                    }
//                    k++;
//                }
//            }
//        }

    }
};

void softmax_csr_parallel(int axis, csr_matrix volatile *A, double volatile *res, int core_id, int nPE) {

    int i, j, k;
    int n_rows, n_cols, n_nnz;
    int *A_row_ptr, *A_col_idx;
    double *A_values;

    n_rows = A->rows;
    n_cols = A->cols;
    n_nnz = A->nnz;

    A_row_ptr = A->row_ptr;
    A_col_idx = A->col_idx;
    A_values = A->values;

    int elr, elr_next, idx;
    double logit, sum, max, emax;

    // For axis zero
    if (axis == 0) {

        for (i = core_id; i < n_rows; i += nPE) {
            elr = A_row_ptr[i];
            elr_next = A_row_ptr[i + 1];
            // Compute the sum
            max = 0;
            for (j = elr; j < elr_next; j++) {
                max = max < A_values[j] ? A_values[j] : max;
            }
            // Compute the sum
            sum = (n_cols - (elr_next - elr)) * my_exp(0.0-max);
            for (j = elr; j < elr_next; j++) {
                sum += my_exp(A_values[j]-max);
            }
            // Compute the Logits
            emax = my_exp(0.0-max);
            logit = (1.0 / sum) * emax;
            for (j = 0; j < 4 * (n_cols >> 2U); j += 4) {
                res[i * n_cols + j] = logit;
                res[i * n_cols + j + 1] = logit;
                res[i * n_cols + j + 2] = logit;
                res[i * n_cols + j + 3] = logit;
            }
            while (j < n_cols) {
                res[i * n_cols + j] = logit;
                j++;
            }
            k = elr;
            while (k < elr_next) {
                idx = A_col_idx[k];
                res[i * n_cols + idx] *= my_exp(A_values[k]);
                k++;
            }
        }

    // For other axes
    } else {

        for (j = core_id; j < n_cols; j += nPE) {
            // Compute the max
            max = 0;
            k = 0;
            while (k < n_nnz) {
                if (A_col_idx[k] == j) {
                    max = max < A_values[k] ? A_values[k] : max;
                }
                k++;
            }
            // Compute the sum
            sum = n_cols;
            k = 0;
            while (k < n_nnz) {
                if (A_col_idx[k] == j) {
                    sum -= 1;
                    sum += my_exp(A_values[k] - max);
                }
                k++;
            }
            // Compute the logits
            logit = (double) (my_exp(0.0 - max); / sum);
            k = A_row_ptr[0];
            for (i = 0; i < n_rows; i++) {
                elr_next = A_row_ptr[i + 1];
                res[i * n_cols + j] = logit;
                while(k < elr_next) {
                    if (A_col_idx[k] == j) {
                        res[i * n_cols + j] = logit * my_exp(A_values[k]);
                    }
                    k++;
                }
            }
        }

    }

};

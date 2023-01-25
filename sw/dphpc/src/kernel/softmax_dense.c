// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#include "softmax_csr.h"
#include "softmax_dense.h"
#include "snrt.h"
#include "printf.h"

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

void softmax_dense_single(int axis, double volatile *A, double volatile *res, int Ncol, int Nrow) {

    int i, j;
    double sum, max, logit;

    // For axis zero
    if (axis == -1) {

        for (i = 0; i < Nrow; i++) {
            max = 0;
            for (j = 0; j < 4 * (Ncol >> 2U); j += 4) {
                double a0 = A[i * Ncol + j];
                double a1 = A[i * Ncol + j + 1];
                double a2 = A[i * Ncol + j + 2];
                double a3 = A[i * Ncol + j + 3];
                max = max < a0 ? a0 : max;
                max = max < a1 ? a1 : max;
                max = max < a2 ? a2 : max;
                max = max < a3 ? a3 : max;
            }
            while (j < Ncol) {
                double a0 = A[i * Ncol + j];
                max = max < a0 ? a0 : max;
                j++;
            }
            sum = 0;
            for (j = 0; j < 4 * (Ncol >> 2U); j += 4) {
                double a0 = A[i * Ncol + j];
                double a1 = A[i * Ncol + j + 1];
                double a2 = A[i * Ncol + j + 2];
                double a3 = A[i * Ncol + j + 3];
                a0 = my_exp(a0 - max);
                a1 = my_exp(a1 - max);
                a2 = my_exp(a2 - max);
                a3 = my_exp(a3 - max);
                sum += a0;
                sum += a1;
                sum += a2;
                sum += a3;
            }
            while (j < Ncol) {
                double a0 = A[i * Ncol + j];
                a0 = my_exp(a0 - max);
                sum += a0;
                j++;
            }
            logit = (double) 1 / sum;
            for (j = 0; j < 4 * (Ncol >> 2U); j += 4) {
                double a0 = A[i * Ncol + j];
                double a1 = A[i * Ncol + j + 1];
                double a2 = A[i * Ncol + j + 2];
                double a3 = A[i * Ncol + j + 3];
                a0 = logit * my_exp(a0 - max);
                a1 = logit * my_exp(a1 - max);
                a2 = logit * my_exp(a2 - max);
                a3 = logit * my_exp(a3 - max);
                res[i * Ncol + j] = a0;
                res[i * Ncol + j + 1] = a1;
                res[i * Ncol + j + 2] = a2;
                res[i * Ncol + j + 3] = a3;
            }
            while (j < Ncol) {
                double a0 = A[i * Ncol + j];
                a0 = logit * my_exp(a0 - max);
                res[i * Ncol + j] = a0;
                j++;
            }
        }

    // For other axes
    } else {

        for (j = 0; j < Ncol; j++) {
            max = 0;
            for (i = 0; i < 4 * (Nrow >> 2U); i += 4) {
                double a0 = A[i * Ncol + j];
                double a1 = A[(i + 1) * Ncol + j];
                double a2 = A[(i + 2) * Ncol + j];
                double a3 = A[(i + 3) * Ncol + j];
                max = max < a0 ? a0 : max;
                max = max < a1 ? a1 : max;
                max = max < a2 ? a2 : max;
                max = max < a3 ? a3 : max;
            }
            while (i < Nrow) {
                double a0 = A[i * Ncol + j];
                max = max < a0 ? a0 : max;
                i++;
            }
            sum = 0;
            for (i = 0; i < 4 * (Nrow >> 2U); i += 4) {
                double a0 = A[i * Ncol + j];
                double a1 = A[(i + 1) * Ncol + j];
                double a2 = A[(i + 2) * Ncol + j];
                double a3 = A[(i + 3) * Ncol + j];
                a0 = my_exp(a0 - max);
                a1 = my_exp(a1 - max);
                a2 = my_exp(a2 - max);
                a3 = my_exp(a3 - max);
                sum += a0;
                sum += a1;
                sum += a2;
                sum += a3;
            }
            while (i < Nrow) {
                double a0 = A[i * Ncol + j];
                a0 = my_exp(a0 - max);
                sum += a0;
                i++;
            }
            logit = (double) 1 / sum;
            for (i = 0; i < 4 * (Nrow >> 2U); i += 4) {
                double a0 = A[i * Ncol + j];
                double a1 = A[(i + 1) * Ncol + j];
                double a2 = A[(i + 2) * Ncol + j];
                double a3 = A[(i + 3) * Ncol + j];
                a0 = logit * my_exp(a0 - max);
                a1 = logit * my_exp(a1 - max);
                a2 = logit * my_exp(a2 - max);
                a3 = logit * my_exp(a3 - max);
                res[i * Ncol + j] = a0;
                res[(i + 1) * Ncol + j] = a1;
                res[(i + 2) * Ncol + j] = a2;
                res[(i + 3) * Ncol + j] = a3;
            }
            while (i < Nrow) {
                double a0 = A[i * Ncol + j];
                a0 = logit * my_exp(a0 - max);
                res[i * Ncol + j] = a0;
                i++;
            }
        }

    }
};

void softmax_dense_parallel(int axis, double volatile *A, double volatile *res, int core_id, int nPE, int Ncol, int Nrow) {

    int i, j;
    double sum, max, logit;

    // For axis zero
    if (axis == -1) {

        for (i = core_id; i < Nrow; i += nPE) {
            max = 0;
            for (j = 0; j < 4 * (Ncol >> 2U); j += 4) {
                double a0 = A[i * Ncol + j];
                double a1 = A[i * Ncol + j + 1];
                double a2 = A[i * Ncol + j + 2];
                double a3 = A[i * Ncol + j + 3];
                max = max < a0 ? a0 : max;
                max = max < a1 ? a1 : max;
                max = max < a2 ? a2 : max;
                max = max < a3 ? a3 : max;
            }
            while (j < Ncol) {
                double a0 = A[i * Ncol + j];
                max = max < a0 ? a0 : max;
                j++;
            }
            sum = 0;
            for (j = 0; j < 4 * (Ncol >> 2U); j += 4) {
                double a0 = A[i * Ncol + j];
                double a1 = A[i * Ncol + j + 1];
                double a2 = A[i * Ncol + j + 2];
                double a3 = A[i * Ncol + j + 3];
                a0 = my_exp(a0 - max);
                a1 = my_exp(a1 - max);
                a2 = my_exp(a2 - max);
                a3 = my_exp(a3 - max);
                sum += a0;
                sum += a1;
                sum += a2;
                sum += a3;
            }
            while (j < Ncol) {
                double a0 = A[i * Ncol + j];
                a0 = my_exp(a0 - max);
                sum += a0;
                j++;
            }
            logit = (double) 1 / sum;
            for (j = 0; j < 4 * (Ncol >> 2U); j += 4) {
                double a0 = A[i * Ncol + j];
                double a1 = A[i * Ncol + j + 1];
                double a2 = A[i * Ncol + j + 2];
                double a3 = A[i * Ncol + j + 3];
                a0 = logit * my_exp(a0 - max);
                a1 = logit * my_exp(a1 - max);
                a2 = logit * my_exp(a2 - max);
                a3 = logit * my_exp(a3 - max);
                res[i * Ncol + j] = a0;
                res[i * Ncol + j + 1] = a1;
                res[i * Ncol + j + 2] = a2;
                res[i * Ncol + j + 3] = a3;
            }
            while (j < Ncol) {
                double a0 = A[i * Ncol + j];
                a0 = logit * my_exp(a0 - max);
                res[i * Ncol + j] = a0;
                j++;
            }
        }

    // For other axes
    } else {

        for (j = core_id; j < Ncol; j += nPE) {
            max = 0;
            for (i = 0; i < 4 * (Nrow >> 2U); i += 4) {
                double a0 = A[i * Ncol + j];
                double a1 = A[(i + 1) * Ncol + j];
                double a2 = A[(i + 2) * Ncol + j];
                double a3 = A[(i + 3) * Ncol + j];
                max = max < a0 ? a0 : max;
                max = max < a1 ? a1 : max;
                max = max < a2 ? a2 : max;
                max = max < a3 ? a3 : max;
            }
            while (i < Nrow) {
                double a0 = A[i * Ncol + j];
                max = max < a0 ? a0 : max;
                i++;
            }
            sum = 0;
            for (i = 0; i < 4 * (Nrow >> 2U); i += 4) {
                double a0 = A[i * Ncol + j];
                double a1 = A[(i + 1) * Ncol + j];
                double a2 = A[(i + 2) * Ncol + j];
                double a3 = A[(i + 3) * Ncol + j];
                a0 = my_exp(a0 - max);
                a1 = my_exp(a1 - max);
                a2 = my_exp(a2 - max);
                a3 = my_exp(a3 - max);
                sum += a0;
                sum += a1;
                sum += a2;
                sum += a3;
            }
            while (i < Nrow) {
                double a0 = A[i * Ncol + j];
                a0 = my_exp(a0 - max);
                sum += a0;
                i++;
            }
            logit = (double) 1 / sum;
            for (i = 0; i < 4 * (Nrow >> 2U); i += 4) {
                double a0 = A[i * Ncol + j];
                double a1 = A[(i + 1) * Ncol + j];
                double a2 = A[(i + 2) * Ncol + j];
                double a3 = A[(i + 3) * Ncol + j];
                a0 = logit * my_exp(a0 - max);
                a1 = logit * my_exp(a1 - max);
                a2 = logit * my_exp(a2 - max);
                a3 = logit * my_exp(a3 - max);
                res[i * Ncol + j] = a0;
                res[(i + 1) * Ncol + j] = a1;
                res[(i + 2) * Ncol + j] = a2;
                res[(i + 3) * Ncol + j] = a3;
            }
            while (i < Nrow) {
                double a0 = A[i * Ncol + j];
                a0 = logit * my_exp(a0 - max);
                res[i * Ncol + j] = a0;
                i++;
            }
        }

    }
};

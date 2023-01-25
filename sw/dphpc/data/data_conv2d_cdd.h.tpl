// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
\
<% def array_to_cstr(array):
    out = '{'
    for a in array.flatten():
        out += '{}, '.format(a)
    out = out[:-2] + '}'
    return out
%> \

#pragma once

#include "matrix_types.h"

#define CHANNELS ${channel_size}
#define NUM_COMP_CORES ${num_proc}

///////////////////////////////////////////////////////////////////////
///////////////////////////     INPUT      ////////////////////////////
///////////////////////////////////////////////////////////////////////

csr_matrix A[${channel_size}];

% for i, m in enumerate(A):
// Data arrays for input matrix A[${i}]
/*
${m.todense()}
*/
double A${i}_data[${m.nnz}] = ${array_to_cstr(m.data)};
int A${i}_indices[${m.nnz}] = ${array_to_cstr(m.indices)};
int A${i}_indptr[${m.shape[1]+1}] = ${array_to_cstr(m.indptr)};
% endfor \

// Array struct for matrix A[${i}]
void assign_A(){
  if (snrt_cluster_core_idx() == 0){
% for i, m in enumerate(A):
    A[${i}] = (csr_matrix){A${i}_data, A${i}_indices, A${i}_indptr, ${m.nnz}, ${m.shape[0]}, ${m.shape[1]}};
% endfor \

  }
}

///////////////////////////////////////////////////////////////////////
///////////////////////////     FILTER      ///////////////////////////
///////////////////////////////////////////////////////////////////////

dense_matrix FILTER_dense[${channel_size}][${channel_size}];

% for i, j in enumerate(FILTER):
% for k, m in enumerate(j):

// Data arrays for input matrix FILTER[${i}][${k}]
/*
${m.todense()}
*/

double FILTER${i}_${k}_data_dense[${filter_dense_elements}] = ${array_to_cstr(m.toarray())};

% endfor
% endfor \

// Array struct for matrix FILTER[${i}][${k}]
void assign_FILTER(){
  if (snrt_cluster_core_idx() == 0){
% for i, j in enumerate(FILTER):
% for k, m in enumerate(j):
     FILTER_dense[${i}][${k}] = (dense_matrix){FILTER${i}_${k}_data_dense, ${m.shape[0]}, ${m.shape[1]}};
% endfor
% endfor \

  }
}

///////////////////////////////////////////////////////////////////////
//////////////////////////     RESULTS      ///////////////////////////
///////////////////////////////////////////////////////////////////////

dense_matrix RES_dense[${channel_size}];

% for i, m in enumerate(RES):
// Data arrays for input matrix RES[${i}]
/*
${m.todense()}
*/

double RES${i}_data_dense[${RES_dense_elements}] = ${array_to_cstr(m.toarray())};
% endfor \

// Array struct for matrix RES[${i}]
void assign_RES(){
  if (snrt_cluster_core_idx() == 0){
% for i, m in enumerate(RES):
    RES_dense[${i}] = (dense_matrix){RES${i}_data_dense, ${m.shape[0]}, ${m.shape[1]}};
% endfor \

  }
}

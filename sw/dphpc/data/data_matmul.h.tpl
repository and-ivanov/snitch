// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
\
<% def array_to_cstr(array):
    if len(array) == 0:
        return '{}'
    out = '{'
    for a in array.flatten():
        out += '{}, '.format(a)
    out = out[:-2] + '}'
    return out
%> \

<%! import math %> \

#pragma once

#include "matrix_types.h"

#define NUM_COMP_CORES ${nproc}
% if measurement:
#define MEASUREMENT
% endif
% for m, m_str in zip([A, B, C], ['A', 'B', 'C']):

// Data arrays for matrix ${m_str}
/*
${m.toarray()}
*/
// Dense representation
double ${m_str}_data_dense[${math.prod(m.shape)}] = ${array_to_cstr(m.toarray())};
double ${m_str}_data[${m.nnz}] = ${array_to_cstr(m.data)};
int ${m_str}_indices[${m.nnz}] = ${array_to_cstr(m.indices)};
int ${m_str}_indptr[${m.shape[1]+1}] = ${array_to_cstr(m.indptr)};

// Array struct for matrix ${m_str}
csr_matrix ${m_str} = {${m_str}_data, ${m_str}_indices, ${m_str}_indptr, ${m.nnz}, ${m.shape[0]}, ${m.shape[1]}};
dense_matrix ${m_str}_dense = {${m_str}_data_dense, ${m.shape[0]}, ${m.shape[1]}};

% endfor \

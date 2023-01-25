// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
\
<% def array_to_cstr(array):
    out = '{'
    for a in array:
        out += '{}, '.format(a)
    out = out[:-2] + '}'
    return out
%> \

#define N (${dim})
#define N_PROC (${nproc})
#define AXIS (${axis})

% for m, m_str in zip([A, C], ['A', 'C']):

// Data arrays for matrix ${m_str}
double ${m_str}[${dim*dim}] = ${array_to_cstr(m)};

% endfor \

// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include "snrt.h"

extern uintptr_t volatile tohost, fromhost;

struct putc_buffer_header {
    int lock;
    uint64_t syscall_mem[8];
};

static char putc_buf[1024];
static struct putc_buffer_header hdr;

// Provide an implementation for putchar.
void snrt_putchar(char character) {
    unsigned core_idx = snrt_cluster_core_idx();
    unsigned core_num = snrt_cluster_core_num();

    int segment_start = (sizeof(putc_buf) * core_idx / core_num);
    int segment_end = (sizeof(putc_buf) * (core_idx + 1) / core_num);
    int segment_size = segment_end - segment_start;

    uint8_t* data_size = (uint8_t*)&putc_buf[segment_start];
    char* data_start = &putc_buf[segment_start + sizeof(*data_size)];
    int data_max_size = segment_size - sizeof(*data_size);
    if (data_max_size > 255) data_max_size = 255;  // prevent uint8_t overflow 
    
    data_start[(*data_size)++] = character;
    if ((*data_size) == data_max_size || character == '\n') {
        int exp = 0;
        while (!__atomic_compare_exchange_n(&hdr.lock, &exp, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE )) { exp = 0; }

        hdr.syscall_mem[0] = 64;  // sys_write
        hdr.syscall_mem[1] = 1;   // file descriptor (1 = stdout)
        hdr.syscall_mem[2] = (uintptr_t) data_start;  // buffer
        hdr.syscall_mem[3] = *data_size;          // length
            
        uintptr_t expected = 0;
        while (!__atomic_compare_exchange_n(&tohost, &expected, (uintptr_t)hdr.syscall_mem, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE )) { expected = 0; }

        while (__atomic_load_n(&fromhost, __ATOMIC_ACQUIRE) == 0) {}
        __atomic_store_n(&fromhost, 0, __ATOMIC_RELEASE);

        __atomic_store_n(&hdr.lock, 0, __ATOMIC_RELEASE);

        *data_size = 0;
    }
}
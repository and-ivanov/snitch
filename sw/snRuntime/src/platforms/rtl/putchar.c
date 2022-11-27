// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include "snrt.h"

extern uintptr_t volatile tohost, fromhost;

#define MAX_THREADS 10
#define BUF_SIZE 1024

struct ThreadBuf {
  uint64_t syscall_mem[8];
  int size;
  char buf[BUF_SIZE];
};

static int lock;
static struct ThreadBuf tb[MAX_THREADS];

#define ATOMIC_CAS(dst, exp, src) do { \
  typeof(exp) e = 0; \
  while (!__atomic_compare_exchange_n( \
    &dst, &e, src, \
    0, __ATOMIC_ACQUIRE, __ATOMIC_RELEASE \
  )) { e = exp; } \
} while (0) 

#define ATOMIC_SET(dst, src) do { \
  __atomic_store_n(&dst, src, __ATOMIC_RELEASE); \
} while (0)

// Provide an implementation for putchar.
void snrt_putchar(char character) {
    int tid = snrt_hartid();
    if (tid > MAX_THREADS) return;

    struct ThreadBuf* ltb = &tb[tid];

    ltb->buf[ltb->size++] = character;
    if (ltb->size == BUF_SIZE || character == '\n') {
        int val = 1;
        while (val) {
          __atomic_exchange(&lock, &val, &val, __ATOMIC_ACQUIRE);
        }

        ltb->syscall_mem[0] = 64;  // sys_write
        ltb->syscall_mem[1] = 1;   // file descriptor (1 = stdout)
        ltb->syscall_mem[2] = (uintptr_t) ltb->buf;  // buffer
        ltb->syscall_mem[3] = ltb->size;          // length
        

        while (__atomic_load_n(&tohost, __ATOMIC_ACQUIRE) != 0) {}
        ATOMIC_SET(tohost, ltb->syscall_mem);

        while (__atomic_load_n(&fromhost, __ATOMIC_RELAXED) == 0) {}
        ATOMIC_SET(fromhost, 0);

        ltb->size = 0;

        __atomic_store_n(&lock, 0, __ATOMIC_RELEASE);
    }
}
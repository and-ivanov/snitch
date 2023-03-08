// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "dm.h"

#include "snrt.h"

//================================================================================
// Settings
//================================================================================

/**
 * @brief Define DM_USE_GLOBAL_CLINT to use the cluster-shared CLINT based SW
 * interrupt system for synchronization. If not defined, the harts use the
 * cluster-local CLINT to syncrhonize which is faster but only works for
 * cluster-local synchronization which is sufficient at the moment since the
 * OpenMP runtime is single cluster only.
 *
 */
// #define DM_USE_GLOBAL_CLINT

/**
 * @brief Number of outstanding transactions to buffer. Each requires
 * sizeof(dm_task_t) bytes
 *
 */
#define DM_TASK_QUEUE_SIZE 4

//================================================================================
// Macros
//================================================================================

#define _dm_mtx_lock() snrt_mutex_lock(&dm_p->mutex)
#define _dm_mtx_release() snrt_mutex_release(&dm_p->mutex)

/**
 * Returns of the dm status call
 */
#define DM_STATUS_COMPLETE_ID 0
#define DM_STATUS_NEXT_ID 1
#define DM_STATUS_BUSY 2
#define DM_STATUS_WOULD_BLOCK 3

//================================================================================
// Types
//================================================================================
typedef struct {
    uint64_t src;
    uint64_t dst;
    uint32_t size;
    uint32_t sstrd;
    uint32_t dstrd;
    uint32_t nreps;
    uint32_t cfg;
    uint32_t twod;
    volatile uint32_t* status;
} dm_task_t;

typedef struct {
    dm_task_t queue[DM_TASK_QUEUE_SIZE];
    volatile uint32_t queue_back;
    volatile uint32_t queue_front_reserved;
    volatile uint32_t queue_front_submitted;
    volatile uint32_t do_exit;
} dm_t;

//================================================================================
// Data
//================================================================================

/**
 * @brief Pointer to the data mover struct in TCDM per thread for faster access
 *
 */
__thread volatile dm_t *dm_p;
/**
 * @brief Pointer to where the DM struct in TCDM is located
 *
 */
static volatile dm_t *volatile dm_p_global;

/**
 * @brief DM core id for wakeup is stored on TLS for performance
 *
 */
__thread uint32_t cluster_dm_core_idx;

//================================================================================
// Declarations
//================================================================================
static void wfi_dm();
static void wake_dm();

//================================================================================
// Debug
//================================================================================
// #define DM_DEBUG_LEVEL 100

#ifdef DM_DEBUG_LEVEL
#include "printf.h"
#define _DM_PRINTF(...)             \
    if (1) {                        \
        printf("[dm] "__VA_ARGS__); \
    }
#define DM_PRINTF(d, ...)        \
    if (DM_DEBUG_LEVEL >= d) {   \
        _DM_PRINTF(__VA_ARGS__); \
    }
#else
#define DM_PRINTF(d, ...)
#endif

//================================================================================
// Publics
//================================================================================
void dm_init(void) {
    cluster_dm_core_idx = snrt_cluster_dm_core_idx();
    // create a data mover instance
    if (snrt_is_dm_core()) {
#ifdef DM_USE_GLOBAL_CLINT
        snrt_interrupt_enable(IRQ_M_SOFT);
#else
        snrt_interrupt_enable(IRQ_M_CLUSTER);
#endif
        dm_p = (dm_t *)snrt_l1alloc(sizeof(dm_t));
        snrt_memset((void *)dm_p, 0, sizeof(dm_t));
        dm_p_global = dm_p;
    } else {
        while (!dm_p_global)
            ;
        dm_p = dm_p_global;
    }
}

void dm_main(void) {
    uint32_t cluster_core_idx = snrt_cluster_core_idx();

    DM_PRINTF(10, "enter main\n");

    while (!dm_p->do_exit) {
        if (dm_p->queue_front_submitted != dm_p->queue_back) {
            // New transaction to issue?
            volatile dm_task_t *t = &dm_p->queue[dm_p->queue_back];
            uint32_t tid;

            if (t->twod) {
                //printf("dm_memcpy2d_async received\n");
                DM_PRINTF(10, "start twod\n");
                tid = __builtin_sdma_start_twod(t->src, t->dst, t->size, t->sstrd,
                                        t->dstrd, t->nreps, t->cfg);
            } else {
                //printf("dm_memcpy_async received\n");
                DM_PRINTF(10, "start oned\n");
                tid = __builtin_sdma_start_oned(t->src, t->dst, t->size, t->cfg);
            }

            while (__builtin_sdma_stat(DM_STATUS_BUSY)) {}
            
            if (t->status) {
                //printf("dm_memcpy_async notification is made\n");
                *(volatile uint32_t*)t->status = 1;  // completed
            } else {
                //printf("dm_memcpy_async notification is not required\n");
            }

            // bump
            dm_p->queue_back = (dm_p->queue_back + 1) % DM_TASK_QUEUE_SIZE;
        } else {
            // queue is empty, go to sleep
            wfi_dm();
        }
    }
    DM_PRINTF(10, "dm: exit\n");
#ifdef DM_USE_GLOBAL_CLINT
    snrt_interrupt_disable(IRQ_M_SOFT);
#else
    snrt_interrupt_disable(IRQ_M_CLUSTER);
#endif
    return;
}

void dm_memcpy_async(void *dest, const void *src, size_t n) {
    dm_memcpy_async_status(dest, src, n, NULL);
}

void dm_memcpy_async_status(void *dest, const void *src, size_t n, volatile uint32_t* status) {
    DM_PRINTF(10, "dm_memcpy_async %#x -> %#x size %d\n", src, dest,
              (uint32_t)n);

    //printf("dm_memcpy_async submitted\n");

    if (status) {
        // init the status as 0 (not completed)
        *status = 0;
    }

    uint32_t s;
    uint32_t s1;
    do {
        s = dm_p->queue_front_reserved;
        s1 = (s + 1) % DM_TASK_QUEUE_SIZE;
    } while (
        // check that we are not reaching back of the queue
        s1 == dm_p->queue_back ||
        // check that the slot we are trying to reserve didn't move
        !__atomic_compare_exchange_n(
            &dm_p->queue_front_reserved, &s, s1,
            0, __ATOMIC_RELAXED, __ATOMIC_RELAXED
        )
    );

    volatile dm_task_t *t = &dm_p->queue[s];
    t->src = (uint64_t)src;
    t->dst = (uint64_t)dest;
    t->size = (uint32_t)n;
    t->twod = 0;
    t->cfg = 0;
    t->status = status;

    // all fields are ready to use, submit
    while (
        !__atomic_compare_exchange_n(
            &dm_p->queue_front_submitted, &s, s1,
            0, __ATOMIC_RELAXED, __ATOMIC_RELAXED
        )
    ) {}

    wake_dm();
}

void dm_memcpy2d_async(uint64_t src, uint64_t dst, uint32_t size,
                       uint32_t sstrd, uint32_t dstrd, uint32_t nreps,
                       uint32_t cfg) {
    dm_memcpy2d_async_status(src, dst, size, sstrd, dstrd, nreps, cfg, NULL);
}

void dm_memcpy2d_async_status(uint64_t src, uint64_t dst, uint32_t size,
                       uint32_t sstrd, uint32_t dstrd, uint32_t nreps,
                       uint32_t cfg, volatile uint32_t* status) {
    DM_PRINTF(10, "dm_memcpy2d_async %#x -> %#x size %d\n", src, dst,
              (uint32_t)size);

    //printf("dm_memcpy2d_async submitted\n");

    if (status) {
        // init the status as 0 (not completed)
        *status = 0;
    }

    uint32_t s;
    uint32_t s1;
    do {
        s = dm_p->queue_front_reserved;
        s1 = (s + 1) % DM_TASK_QUEUE_SIZE;
    } while (
        // check that we are not reaching back of the queue
        s1 == dm_p->queue_back ||
        // check that the slot we are trying to reserve didn't move
        !__atomic_compare_exchange_n(
            &dm_p->queue_front_reserved, &s, s1,
            0, __ATOMIC_RELAXED, __ATOMIC_RELAXED
        )
    );

    // insert
    volatile dm_task_t *t = &dm_p->queue[s];
    t->src = src;
    t->dst = dst;
    t->size = size;
    t->sstrd = sstrd;
    t->dstrd = dstrd;
    t->nreps = nreps;
    t->twod = 1;
    t->cfg = cfg;
    t->status = status;

    // all fields are ready to use, submit
    while (
        !__atomic_compare_exchange_n(
            &dm_p->queue_front_submitted, &s, s1,
            0, __ATOMIC_RELAXED, __ATOMIC_RELAXED
        )
    ) {}

    wake_dm();
}

void dm_wait() {
    while (dm_p->queue_front_submitted != dm_p->queue_back) {
        wake_dm();
    }
}

void dm_wait_status(volatile uint32_t* status) {
    while (!*status) {
        wake_dm();
    }
}

void dm_exit(void) {
    dm_p->do_exit = 1;
    wake_dm();
}

//================================================================================
// private
//================================================================================

#ifdef DM_USE_GLOBAL_CLINT
static void wfi_dm() {
    snrt_int_sw_poll();
}
static void wake_dm(void) {
    uint32_t basehart = snrt_cluster_core_base_hartid();
    snrt_int_sw_set(basehart + cluster_dm_core_idx);
}
#else
static void wfi_dm() {
    snrt_wfi();
    snrt_int_cluster_clr(1 << snrt_cluster_dm_core_idx());
}
static void wake_dm() {
    snrt_int_cluster_set(1 << snrt_cluster_dm_core_idx());
}
#endif  // #ifdef DM_USE_GLOBAL_CLINT

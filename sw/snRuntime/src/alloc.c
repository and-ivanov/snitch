// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include "debug.h"
#include "snrt.h"
#include "team.h"
#include "nano_malloc.h"

#define ALIGN_UP(addr, size) (((addr) + (size)-1) & ~((size)-1))
#define ALIGN_DOWN(addr, size) ((addr) & ~((size)-1))

#define MIN_CHUNK_SIZE 8

/**
 * @brief Allocate a chunk of memory in the L1 memory
 *
 * @param size number of bytes to allocate
 * @return pointer to the allocated memory
 */
void *snrt_l1alloc(size_t size) {
    struct snrt_allocator_inst *alloc = &snrt_current_team()->allocator.l1;
    // make sure that returned pointer is aligned by allocating slightly more
    // and keep the nonaligned memory start offset one byte before the aligned address
    uint32_t align = 8;
    uint32_t align_mask = align - 1;
    char* ptr = (char*) alloc_malloc(alloc->base, size + align);
    char* aligned_ptr = (char*)((uint32_t)(ptr + align) & (~align_mask));
    uint8_t align_offset = (uint8_t)(aligned_ptr - ptr);
    *(aligned_ptr - 1) = align_offset;
    return aligned_ptr;
}

/**
 * @brief Free a chunk of memory in the L1 memory
 *
 * @param ptr pointer to the allocated memory
 */
void snrt_l1free(void* ptr) {
    uint8_t align_offset = *((uint8_t*)ptr - 1);
    alloc_free((char*)ptr - align_offset);
}

/**
 * @brief Allocate a chunk of memory in the L3 memory
 * @details This currently does not support free-ing of memory
 *
 * @param size number of bytes to allocate
 * @return pointer to the allocated memory
 */
void *snrt_l3alloc(size_t size) {
    struct snrt_allocator_inst *alloc = &snrt_current_team()->allocator.l3;

    // TODO: L3 alloc size check

    void *ret = (void *)alloc->next;
    alloc->next += ALIGN_UP(size, MIN_CHUNK_SIZE);
    return ret;
}

/**
 * @brief Init the allocator
 * @details
 *
 * @param snrt_team_root pointer to the team structure
 * @param l3off Number of bytes to skip on _edram before starting allocator
 */
void snrt_alloc_init(struct snrt_team_root *team) {
    // Allocator in L1 TCDM memory
    team->allocator.l1.base = (uint32_t)team->cluster_mem.start;
    team->allocator.l1.size =
        (uint32_t)(team->cluster_mem.end - team->cluster_mem.start);
    team->allocator.l1.next = team->allocator.l1.base;

    alloc_init(
        (void*)team->cluster_mem.start,
        team->cluster_mem.end - team->cluster_mem.start
    );

    // Allocator in L3 shared memory
    extern char _end;
    team->allocator.l3.base =
        ALIGN_UP((uint32_t)&_end, MIN_CHUNK_SIZE);
    ;
    team->allocator.l3.size = 0;
    team->allocator.l3.next = team->allocator.l3.base;
}

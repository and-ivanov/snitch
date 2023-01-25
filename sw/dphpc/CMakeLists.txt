# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

# Allow benchmark to be built as a standalone library.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
    set(CMAKE_TOOLCHAIN_FILE toolchain-llvm CACHE STRING "Toolchain to use")
    set(SNITCH_SIMULATOR ${CMAKE_CURRENT_SOURCE_DIR}/../../hw/system/snitch_cluster/bin/snitch_cluster.vlt CACHE PATH "")

    project(DPHPC LANGUAGES C ASM)
    include(SnitchUtilities)

    add_compile_options(-O3 -g -ffunction-sections)

    # Build the runtime.
    add_subdirectory(../snRuntime snRuntime)
endif()

macro(add_snitch_dphpc_executable name)
    add_executable(${ARGV})
    target_link_libraries(${name} ${SNITCH_RUNTIME})
    target_link_options(${name} PRIVATE "SHELL:-T ${LINKER_SCRIPT}")
    string(REGEX MATCH "^[^_]+" KERNEL ${name})
    if (KERNEL STREQUAL "matmul")
        string(REGEX MATCHALL "_[a-z]+" FORMATS ${name})
        list(GET FORMATS 0 FORMAT_A)
        list(GET FORMATS 1 FORMAT_B)
        string(REGEX MATCH "[a-z]+" FORMAT_A_STRIPPED ${FORMAT_A})
        string(REGEX MATCH "[a-z]+" FORMAT_B_STRIPPED ${FORMAT_B})
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_LIST_DIR}/data/data_${KERNEL}.h
            COMMAND ./data/data_gen_matmul_csr.py --outdir ${CMAKE_CURRENT_LIST_DIR}/data --tpl ${CMAKE_CURRENT_LIST_DIR}/data/data_matmul.h.tpl
            DEPENDS data/data_gen_matmul_csr.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        )
    elseif (KERNEL STREQUAL "softmax")
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_LIST_DIR}/data/data_${name}.h
            COMMAND ./data/data_gen_${name}.py --outdir ${CMAKE_CURRENT_LIST_DIR}/data
            DEPENDS data/data_gen_${name}.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        )
    elseif (KERNEL STREQUAL "conv2d")
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_LIST_DIR}/data/data_${name}.h
            COMMAND ./data/data_gen_conv2d.py --outdir ${CMAKE_CURRENT_LIST_DIR}/data
            DEPENDS data/data_gen_conv2d.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        )
    endif()
    add_custom_command(
        TARGET ${name}
        POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -dhS $<TARGET_FILE:${name}> > $<TARGET_FILE:${name}>.s)
    # Run target for banshee
    if (SNITCH_RUNTIME STREQUAL "snRuntime-banshee")
        add_custom_target( run-banshee-${name}
            COMMAND ${SNITCH_BANSHEE} --no-opt-llvm --no-opt-jit --configuration ${CMAKE_CURRENT_LIST_DIR}/../banshee/config/snitch_cluster.yaml --trace $<TARGET_FILE:${name}> > $<TARGET_FILE:${name}>.trace
            COMMAND cat $<TARGET_FILE:${name}>.trace | ${SPIKE_DASM} > $<TARGET_FILE:${name}>.trace.txt
            COMMAND awk -F\" \" '{print>\"${name}\"$$3\".txt\"}' $<TARGET_FILE:${name}>.trace.txt
            DEPENDS $<TARGET_FILE:${name}>)
    endif()
    # Run target for RTL simulator
    if (SNITCH_RUNTIME STREQUAL "snRuntime-cluster")
        add_custom_target( run-rtl-${name}
            COMMAND ${SNITCH_SIMULATOR} $<TARGET_FILE:${name}>
            COMMAND for f in logs/trace_hart_*.dasm\; do ${SPIKE_DASM} < $$f | ${PYTHON} ${SNRUNTIME_SRC_DIR}/../../util/gen_trace.py > $$\(echo $$f | sed 's/\\.dasm/\\.txt/'\)\; done
            DEPENDS $<TARGET_FILE:${name}>)
    endif()
    target_link_libraries(${name} kernels utils)
endmacro()

enable_testing()

include_directories(include data src/utils src/kernels)
include_directories(${SNRUNTIME_INCLUDE_DIRS})

add_library(utils src/utils/utils.c)
add_library(kernels src/kernel/matmul_csr.c src/kernel/softmax_csr.c src/kernel/softmax_dense.c src/kernel/conv2d_csr.c)

target_link_libraries(kernels ${SNITCH_RUNTIME} utils)
target_link_libraries(utils ${SNITCH_RUNTIME})

add_snitch_dphpc_executable(matmul_csr_csr src/main_matmul_csr_csr.c data/data_matmul.h)
add_snitch_dphpc_executable(matmul_csr_csr_to_dense src/main_matmul_csr_csr_to_dense.c data/data_matmul.h)
add_snitch_dphpc_executable(matmul_csr_dense src/main_matmul_csr_dense.c data/data_matmul.h)
add_snitch_dphpc_executable(matmul_csr_dense_to_dense src/main_matmul_csr_dense_to_dense.c data/data_matmul.h)
add_snitch_dphpc_executable(matmul_dense_dense src/main_matmul_dense_dense.c data/data_matmul.h)
add_snitch_dphpc_executable(softmax_csr src/main_softmax_csr.c data/data_softmax_csr.h)
add_snitch_dphpc_executable(softmax_dense src/main_softmax_dense.c data/data_softmax_dense.h)
add_snitch_dphpc_executable(conv2d_csr_csr_csr       src/main_conv2d_csr_csr_csr.c       data/data_conv2d_csr_csr_csr.h      )
add_snitch_dphpc_executable(conv2d_csr_csr_dense     src/main_conv2d_csr_csr_dense.c     data/data_conv2d_csr_csr_dense.h    )
add_snitch_dphpc_executable(conv2d_csr_dense_csr     src/main_conv2d_csr_dense_csr.c     data/data_conv2d_csr_dense_csr.h    )
add_snitch_dphpc_executable(conv2d_csr_dense_dense   src/main_conv2d_csr_dense_dense.c   data/data_conv2d_csr_dense_dense.h  )
add_snitch_dphpc_executable(conv2d_dense_dense_dense src/main_conv2d_dense_dense_dense.c data/data_conv2d_dense_dense_dense.h)
add_snitch_dphpc_executable(conv2d_dense_csr_dense src/main_conv2d_dense_csr_dense.c data/data_conv2d_dense_csr_dense.h)
add_snitch_dphpc_executable(conv2d_dense_csrr_dense src/main_conv2d_dense_csrr_dense.c data/data_conv2d_dense_csrr_dense.h)

set(SNITCH_TEST_PREFIX snDPHPC-)

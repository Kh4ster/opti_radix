#pragma once

#include "cuda_tools/host_shared_ptr.cuh"
#include <benchmark/benchmark.h>

template <typename T, typename FUNC>
void check_buffer(cuda_tools::host_shared_ptr<T> buffer,
                  FUNC func,
                  benchmark::State& st);

// Compare device data of buffer with host data of expected
// Retriving data host side in buffer before already performed
template <typename T>
void check_buffer(cuda_tools::host_shared_ptr<T> buffer,
                  cuda_tools::host_shared_ptr<T> expected,
                  benchmark::State& st);

#include "test_helpers.hxx"
#include "test_helpers.hh"

#include "cuda_tools/host_shared_ptr.cuh"

#include <gtest/gtest.h>

void check_buffer(cuda_tools::host_shared_ptr<int> buffer)
{
    int *host_buffer = buffer.download();
    for (std::size_t i = 0; i < buffer.size_; ++i)
        ASSERT_EQ(host_buffer[i], 0);
}
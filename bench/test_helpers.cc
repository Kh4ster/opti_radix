#include "test_helpers.hh"

#include "cuda_tools/host_shared_ptr.cuh"

#include <gtest/gtest.h>

void check_buffer(cuda_tools::host_shared_ptr<int> buffer)
{
    //for (int i = 0; i < buffer.size_; ++i)
    //    ASSERT_EQ(buffer[i], 0);
}
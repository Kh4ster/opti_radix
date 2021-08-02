#include "test_helpers.hh"

#include "cuda_tools/host_shared_ptr.cuh"

#include <algorithm>
#include <execution>
#include <gtest/gtest.h>

template <typename FUNC>
void check_buffer(cuda_tools::host_shared_ptr<int> buffer, FUNC func)
{
    int* host_buffer = buffer.download();

    // All values should be equal to val + 1 (done in kernel)
    if (!std::all_of(std::execution::par_unseq,
                     host_buffer,
                     host_buffer + buffer.size_,
                     func))
    {
        ASSERT_TRUE(false);
    }
}
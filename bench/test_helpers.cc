#include "test_helpers.hh"

#include "cuda_tools/host_shared_ptr.cuh"

#include <algorithm>
#include <execution>
#include <gtest/gtest.h>
#include <vector>

void check_buffer(cuda_tools::host_shared_ptr<int> buffer, const int val)
{
    int* host_buffer = buffer.download();
    std::vector<int> to_check(host_buffer, host_buffer + buffer.size_);

    // All values should be equal to val + 1 (done in kernel)
    auto lambda = [val](int i) { return i == val + 1; };
    if (!std::all_of(std::execution::par_unseq,
                     to_check.cbegin(),
                     to_check.cend(),
                     lambda))
    {
        ASSERT_TRUE(false);
    }
}
#include "test_helpers.hh"

#include "cuda_tools/host_shared_ptr.cuh"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <execution>
#include <gtest/gtest.h>

template <typename T, typename FUNC>
void check_buffer(cuda_tools::host_shared_ptr<int> buffer,
                  FUNC func,
                  benchmark::State& st)
{
    int* host_buffer = buffer.download();

    // All values should be equal to val + 1 (done in kernel)
    if (!std::all_of(std::execution::par_unseq,
                     host_buffer,
                     host_buffer + buffer.size_,
                     func))
    {
        st.SkipWithError("Failed test");
        ASSERT_TRUE(false);
    }
}

template <typename T>
void check_buffer(cuda_tools::host_shared_ptr<T> buffer,
                  cuda_tools::host_shared_ptr<T> expected,
                  benchmark::State& st)
{
    T* host_buffer = buffer.download();

    if (!std::equal(std::execution::par_unseq,
                    host_buffer,
                    host_buffer + buffer.size_,
                    expected.host_data_))
    {
        auto [first, second] = std::mismatch(std::execution::par_unseq,
                                             host_buffer,
                                             host_buffer + buffer.size_,
                                             expected.host_data_);
        std::cout << "Error at " << first - host_buffer << ": " << *first << " "
                  << *second << std::endl;
        st.SkipWithError("Failed test");
        ASSERT_FALSE(true);
    }
}
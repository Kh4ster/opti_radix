#pragma once

#include <device_launch_parameters.h>

#include "host_shared_ptr.cuh"

// This class purpose is to avoid passing host data to kernel
namespace cuda_tools
{

template <typename T>
struct host_shared_ptr;
template <typename T>
struct device_buffer
{
    explicit device_buffer() = default;
    explicit device_buffer(host_shared_ptr<T>& ptr);
    ~device_buffer() = default;

    __device__ T* get() const noexcept;
    __device__ T* get() noexcept;

    __device__ T& operator[](std::ptrdiff_t idx) const;
    __device__ T& operator[](std::ptrdiff_t idx);

    T* __restrict__ data_ = nullptr;
    int size_ = 0;
};

} // namespace cuda_tools
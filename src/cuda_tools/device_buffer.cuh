#pragma once

#include "host_shared_ptr.cuh"

#include <device_launch_parameters.h>

namespace cuda_tools
{
template <typename T>
struct device_buffer
{
    explicit device_buffer() = default;
    explicit device_buffer(host_shared_ptr<T>& ptr);
    ~device_buffer() = default;
    
    __device__
    T* get() const noexcept;
    __device__
    T* get() noexcept;

    __device__
    T& operator[](std::ptrdiff_t idx) const;
    __device__
    T& operator[](std::ptrdiff_t idx);

    T* __restrict__ data_ = nullptr;
    std::size_t size_ = 0; 
};

} // namespace cuda_tools
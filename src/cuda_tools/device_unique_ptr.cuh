#pragma once

#include "host_unique_ptr.cuh"

#include <device_launch_parameters.h>

namespace cuda_tools
{
template <typename T>
struct device_unique_ptr
{
    explicit device_unique_ptr() = default;
    explicit device_unique_ptr(host_unique_ptr<T> ptr);
    
    // No destructor free as it would cause desctruction in kernel
    ~device_unique_ptr() = default;
    
    __device__
    T* get() const noexcept;
    __device__
    T* get() noexcept;

    __device__
    T& operator[](std::ptrdiff_t idx) const;
    __device__
    T& operator[](std::ptrdiff_t idx);

    T* data_ = nullptr;
    std::size_t size_ = 0; 
};

} // namespace cuda_tools
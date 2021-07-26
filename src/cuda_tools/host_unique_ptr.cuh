#pragma once

#include <cstddef>

namespace cuda_tools
{
template <typename T>
struct host_unique_ptr
{
    explicit host_unique_ptr() = default;

    explicit host_unique_ptr(std::size_t size);
    
    // No destructor free as it would cause desctruction in kernel
    ~host_unique_ptr() = default;
    
    void allocate(std::size_t size);

    void free();

    T* data_ = nullptr;
    std::size_t size_ = 0; 
};

} // namespace cuda_tools
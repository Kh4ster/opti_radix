#include "device_buffer.cuh"

#include <cuda_runtime.h>

#include "cuda_error_checking.cuh"
#include "template_generator.hh"

namespace cuda_tools
{

template_generation(device_buffer);

template <typename T>
device_buffer<T>::device_buffer(host_shared_ptr<T>& ptr) : data_(ptr.data_), size_(ptr.size_)
{}

template <typename T>
__device__
inline T* device_buffer<T>::get() const noexcept
{
    return data_;
}

template <typename T>
__device__
inline T* device_buffer<T>::get() noexcept
{
    return data_;
}

template <typename T>
__device__
inline T& device_buffer<T>::operator[](std::ptrdiff_t idx) const
{
    return get()[idx];
}

template <typename T>
__device__
inline T& device_buffer<T>::operator[](std::ptrdiff_t idx)
{
    return get()[idx];
}

}
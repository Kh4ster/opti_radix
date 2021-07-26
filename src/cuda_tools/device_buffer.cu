#include "device_buffer.cuh"

#include <cuda_runtime.h>

#include "cuda_error_checking.cuh"

namespace cuda_tools
{

template class device_buffer<short>;
template class device_buffer<int>;
template class device_buffer<long>;
template class device_buffer<long long>;

template class device_buffer<unsigned short>;
template class device_buffer<unsigned int>;
template class device_buffer<unsigned long>;
template class device_buffer<unsigned long long>;

template class device_buffer<float>;
template class device_buffer<double>;

template <typename T>
device_buffer<T>::device_buffer(host_shared_ptr<T>& ptr) : data_(ptr.data_), size_(ptr.size_)
{}

template <typename T>
__device__
T* device_buffer<T>::get() const noexcept
{
    return data_;
}

template <typename T>
__device__
T* device_buffer<T>::get() noexcept
{
    return data_;
}

template <typename T>
__device__
T& device_buffer<T>::operator[](std::ptrdiff_t idx) const
{
    return get()[idx];
}

template <typename T>
__device__
T& device_buffer<T>::operator[](std::ptrdiff_t idx)
{
    return get()[idx];
}

}
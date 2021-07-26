#include "device_unique_ptr.cuh"

#include <cuda_runtime.h>

#include "cuda_error_checking.cuh"

namespace cuda_tools
{

template class device_unique_ptr<short>;
template class device_unique_ptr<int>;
template class device_unique_ptr<long>;
template class device_unique_ptr<long long>;

template class device_unique_ptr<unsigned short>;
template class device_unique_ptr<unsigned int>;
template class device_unique_ptr<unsigned long>;
template class device_unique_ptr<unsigned long long>;

template class device_unique_ptr<float>;
template class device_unique_ptr<double>;

template <typename T>
device_unique_ptr<T>::device_unique_ptr(host_unique_ptr<T> ptr) : data_(ptr.data_), size_(ptr.size_)
{

}

template <typename T>
__device__
T* device_unique_ptr<T>::get() const noexcept
{
    return data_;
}

template <typename T>
__device__
T* device_unique_ptr<T>::get() noexcept
{
    return data_;
}

template <typename T>
__device__
T& device_unique_ptr<T>::operator[](std::ptrdiff_t idx) const
{
    return get()[idx];
}

template <typename T>
__device__
T& device_unique_ptr<T>::operator[](std::ptrdiff_t idx)
{
    return get()[idx];
}

}
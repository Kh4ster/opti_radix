#include "host_shared_ptr.cuh"

#include <cuda_runtime.h>

#include "cuda_error_checking.cuh"

namespace cuda_tools
{

template class host_shared_ptr<short>;
template class host_shared_ptr<int>;
template class host_shared_ptr<long>;
template class host_shared_ptr<long long>;

template class host_shared_ptr<unsigned short>;
template class host_shared_ptr<unsigned int>;
template class host_shared_ptr<unsigned long>;
template class host_shared_ptr<unsigned long long>;

template class host_shared_ptr<float>;
template class host_shared_ptr<double>;

template <typename T>
__host__
void host_shared_ptr<T>::allocate(std::size_t size)
{
    cuda_safe_call(cudaMalloc(&data_, sizeof(T) * size));
}

template <typename T>
host_shared_ptr<T>::host_shared_ptr(std::size_t size) : size_(size)
{
    allocate(size);
}

template <typename T>
host_shared_ptr<T>::host_shared_ptr(host_shared_ptr<T>&& ptr) : data_(ptr.data_), size_(ptr.size_), counter_(ptr.counter_ + 1)
{}

template <typename T>
host_shared_ptr<T>::host_shared_ptr(host_shared_ptr<T>& ptr) : data_(ptr.data_), size_(ptr.size_), counter_(ptr.counter_ + 1)
{}

template <typename T>
host_shared_ptr<T>& host_shared_ptr<T>::operator=(host_shared_ptr<T>&& r)
{
    data_ = r.data_;
    size_ = r.size_;
    counter_ = r.counter_ + 1;
    return *this;
}

template <typename T>
host_shared_ptr<T>::~host_shared_ptr()
{
    if (--counter_ == 0)
        cuda_safe_call(cudaFree(data_));
}

} // namespace cuda_tools
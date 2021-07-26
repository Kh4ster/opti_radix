#include "host_shared_ptr.cuh"

#include <cuda_runtime.h>

#include "cuda_error_checking.cuh"
#include "template_generator.hh"

namespace cuda_tools
{

template_generation(host_shared_ptr);

template <typename T>
__host__
void host_shared_ptr<T>::allocate(std::size_t size)
{
    cuda_safe_call(cudaMalloc((void**)&data_, sizeof(T) * size));
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
    {
        cuda_safe_call(cudaFree(data_));
        if (host_data_ != nullptr)
            delete[] host_data_;
    }
}
template <typename T>
T* host_shared_ptr<T>::download()
{
    if (data_ != nullptr)
    {
        if (host_data_ == nullptr)
            host_data_ = new T[size_];
        cuda_safe_call(cudaMemcpy(host_data_, data_, sizeof(T) * size_, cudaMemcpyDeviceToHost));
    }
    return host_data_;
}

} // namespace cuda_tools
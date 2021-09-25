#include "host_shared_ptr.cuh"
#include "cuda_tools/device_buffer.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <execution>

#include "cuda_error_checking.cuh"
#include "template_generator.hh"

namespace cuda_tools
{

template_generation(host_shared_ptr);

template <typename T>
void host_shared_ptr<T>::device_allocate(std::size_t size)
{
    cuda_safe_call(cudaMalloc((void**)&data_, sizeof(T) * size));
}

template <typename T>
host_shared_ptr<T>::host_shared_ptr(std::size_t size) : size_(size)
{
    device_allocate(size);
}

template <typename T>
host_shared_ptr<T>::host_shared_ptr(host_shared_ptr<T>&& ptr) : data_(ptr.data_), host_data_(ptr.host_data_), size_(ptr.size_), counter_(ptr.counter_ + 1)
{}

template <typename T>
host_shared_ptr<T>::host_shared_ptr(host_shared_ptr<T>& ptr) : data_(ptr.data_), host_data_(ptr.host_data_), size_(ptr.size_), counter_(ptr.counter_ + 1)
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
inline T& host_shared_ptr<T>::operator[](std::ptrdiff_t idx) const noexcept
{
    return host_data_[idx];
}

template <typename T>
inline T& host_shared_ptr<T>::operator[](std::ptrdiff_t idx) noexcept
{
    return host_data_[idx];
}

template <typename T>
T* host_shared_ptr<T>::download()
{
    if (host_data_ == nullptr)
        host_allocate();
    cuda_safe_call(cudaMemcpy(host_data_, data_, sizeof(T) * size_, cudaMemcpyDeviceToHost));
    return host_data_;
}

template <typename T>
void host_shared_ptr<T>::upload()
{
    cuda_safe_call(cudaMemcpy(data_, host_data_, sizeof(T) * size_, cudaMemcpyHostToDevice));
}

template <typename T, typename FUNC>
__global__
static void kernel_fill(cuda_tools::device_buffer<T> buffer, FUNC func)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < buffer.size_)
        buffer[index] = func();
}

template <typename T>
void host_shared_ptr<T>::device_fill(const T val)
{
    constexpr int TILE_WIDTH  = 64;
    constexpr int TILE_HEIGHT = 1;

    cuda_tools::device_buffer<T> device_buffer(*this);

    auto lambda = [val] __device__ { return val; };

    const int gx = (this->size_ + TILE_WIDTH - 1) / TILE_WIDTH;
    const int gy = 1;

    const dim3 block(TILE_WIDTH, TILE_HEIGHT);
    const dim3 grid(gx, gy);

    kernel_fill<T><<<grid, block>>>(device_buffer, lambda);
    kernel_check_error();

    cudaDeviceSynchronize();
}

template <typename T>
void host_shared_ptr<T>::host_allocate()
{
    host_allocate(size_);
}

template <typename T>
void host_shared_ptr<T>::host_allocate(std::size_t size)
{
    host_data_ = new T[size];
    size_ = size;
}

template <typename T>
void host_shared_ptr<T>::host_fill(const T val)
{
    std::transform(std::execution::par_unseq,
                   host_data_,
                   host_data_ + size_,
                   host_data_,
                   [val]([[maybe_unused]]T arg){return val;});
}

} // namespace cuda_tools
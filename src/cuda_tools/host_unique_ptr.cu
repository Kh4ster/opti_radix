#include "host_unique_ptr.cuh"

#include <cuda_runtime.h>

#include "cuda_error_checking.cuh"

namespace cuda_tools
{

template class host_unique_ptr<short>;
template class host_unique_ptr<int>;
template class host_unique_ptr<long>;
template class host_unique_ptr<long long>;

template class host_unique_ptr<unsigned short>;
template class host_unique_ptr<unsigned int>;
template class host_unique_ptr<unsigned long>;
template class host_unique_ptr<unsigned long long>;

template class host_unique_ptr<float>;
template class host_unique_ptr<double>;

template <typename T>
__host__
void host_unique_ptr<T>::allocate(std::size_t size)
{
    cuda_safe_call(cudaMalloc(&data_, sizeof(T) * size));
}

template <typename T>
host_unique_ptr<T>::host_unique_ptr(std::size_t size) : size_(size)
{
    allocate(size);
}

template <typename T>
void host_unique_ptr<T>::free()
{
    cuda_safe_call(cudaFree(data_));
}

} // namespace cuda_tools
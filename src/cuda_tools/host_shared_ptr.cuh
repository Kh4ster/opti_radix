#pragma once

#include <cstddef>
#include <functional>

namespace cuda_tools
{
template <typename T>
struct host_shared_ptr
{
    host_shared_ptr() = default;
    host_shared_ptr(std::size_t size);
    host_shared_ptr(host_shared_ptr<T>&& ptr);
    host_shared_ptr(host_shared_ptr<T>& ptr);
    host_shared_ptr& operator=(host_shared_ptr<T>&& r);

    ~host_shared_ptr();

    T& operator[](std::ptrdiff_t idx) const noexcept;
    T& operator[](std::ptrdiff_t idx) noexcept;

    void host_allocate(std::size_t size);
    void host_allocate();
    void host_fill(const T val);
    void host_fill(std::function<T()> func);
    void host_map(std::function<T(T arg)> func);
    void device_allocate(std::size_t size);
    void device_fill(const T val);

    // Transfer & get device data to host data
    T* download();
    // Transfer host data to device data
    void upload();

    T* __restrict__ data_ = nullptr;
    T* __restrict__ host_data_ = nullptr;
    int size_ = 0;
    int counter_ = 1;
};

} // namespace cuda_tools
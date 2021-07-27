#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/device_buffer.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

__global__ void kernel(cuda_tools::device_buffer<int> buffer)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < buffer.size_)
        buffer[index] = 0;
}

void to_bench_single(cuda_tools::host_shared_ptr<int> buffer)
{
    cuda_tools::device_buffer<int> device_buffer(buffer);

    constexpr int TILE_WIDTH = 64;
    constexpr int TILE_HEIGHT = 1;

    const int gx = (buffer.size_ + TILE_WIDTH - 1) / TILE_WIDTH;
    const int gy = 1;

    const dim3 block(TILE_WIDTH, TILE_HEIGHT);
    const dim3 grid(gx, gy);

    kernel<<<grid, block>>>(device_buffer);
    kernel_check_error();

    cudaDeviceSynchronize();
}

void to_bench_multiple(cuda_tools::host_shared_ptr<int> buffer,
              int tile_width,
              int tile_height)
{
    cuda_tools::device_buffer<int> device_buffer(buffer);

    const int gx = (buffer.size_ + tile_width - 1) / tile_width;
    const int gy = 1;

    const dim3 block(tile_width, tile_height);
    const dim3 grid(gx, gy);

    kernel<<<grid, block>>>(device_buffer);
    kernel_check_error();

    cudaDeviceSynchronize();
}
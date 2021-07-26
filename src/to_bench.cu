#include "to_bench.cuh"

#include "cuda_tools/host_unique_ptr.cuh"
#include "cuda_tools/device_unique_ptr.cuh"
#include "cuda_tools/cuda_error_checking.cuh"

__global__
void kernel(cuda_tools::device_unique_ptr<int> buffer)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < buffer.size_)
        buffer[index] = 0;
}


void to_bench(cuda_tools::host_unique_ptr<int> buffer)
{
    cuda_tools::device_unique_ptr<int> device_buffer(buffer);

    constexpr int TILE_WIDTH  = 64;
    constexpr int TILE_HEIGHT = 1;

    const int gx             = (buffer.size_ + TILE_WIDTH - 1) / TILE_WIDTH;
    const int gy             = 1;
    
    const dim3 block(TILE_WIDTH, TILE_HEIGHT);
    const dim3 grid(gx, gy);

    kernel<<<grid, block>>>(device_buffer);
    kernel_check_error();

    cudaDeviceSynchronize();
}
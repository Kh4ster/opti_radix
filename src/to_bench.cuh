#pragma once

#include "cuda_tools/host_shared_ptr.cuh"

void to_bench(cuda_tools::host_shared_ptr<int> buffer);
void to_bench(cuda_tools::host_shared_ptr<int> buffer,
              int tile_width,
              int tile_height);
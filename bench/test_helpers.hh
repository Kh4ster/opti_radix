#pragma once

#include "cuda_tools/host_shared_ptr.cuh"

void check_buffer(cuda_tools::host_shared_ptr<int> buffer, const int val);
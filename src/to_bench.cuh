#pragma once

#include "cuda_tools/host_shared_ptr.cuh"

class Chrono
{
  public:
    virtual void PauseTiming() = 0;
    virtual void ResumeTiming() = 0;
    virtual void SetIterationTime(double seconds) = 0;
};

template <typename T = unsigned char,
          int TILE_WIDTH = 256,
          int TILE_HEIGHT = 1,
          int NB_BITS = 1>
void radix_sort_basic(cuda_tools::host_shared_ptr<T> in,
                      cuda_tools::host_shared_ptr<T> out,
                      Chrono* chrono);

#include "to_bench.cuhxx"
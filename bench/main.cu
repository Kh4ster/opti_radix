#include "cuda_tools/host_shared_ptr.cuh"
#include "test_helpers.hh"
#include "to_bench.cuh"

#include <benchmark/benchmark.h>
#include <cstdlib>

struct GBenchmarkChrono : public Chrono
{
    GBenchmarkChrono(benchmark::State* state)
        : m_st{state}
    {
    }

    void ResumeTiming() final { m_st->ResumeTiming(); }
    void PauseTiming() final { m_st->PauseTiming(); }
    void SetIterationTime(double seconds) { m_st->SetIterationTime(seconds); }

  private:
    benchmark::State* m_st;
};

class Fixture : public benchmark::Fixture
{
  public:
    static bool no_check;

    template <typename T, typename FUNC, typename... Args>
    void bench_gpu(benchmark::State& st,
                   FUNC callback,
                   std::size_t size,
                   Args&&... args)
    {
        GBenchmarkChrono chrono = {&st};

        cuda_tools::host_shared_ptr<T> in(size);
        in.host_fill([]() { return rand() % 1000; });
        in.upload();
        cuda_tools::host_shared_ptr<T> out(size);

        cuda_tools::host_shared_ptr<T> ref = in.copy();
        std::sort(ref.host_data_, ref.host_data_ + ref.size_);

        for (auto _ : st)
        {
            callback(in, out, &chrono);
        }

        st.SetBytesProcessed(int64_t(st.iterations()) *
                             int64_t(size * sizeof(int)));

        if (!no_check)
            check_buffer(out, ref, st);
    }

    template <typename T, typename... Args>
    void bench_cpu(benchmark::State& st, std::size_t size, Args&&... args)
    {
        cuda_tools::host_shared_ptr<T> in(size);
        in.host_fill([]() { return rand() % 1000; });
        in.upload();

        for (auto _ : st)
            std::sort(in.host_data_, in.host_data_ + in.size_);

        st.SetBytesProcessed(int64_t(st.iterations()) *
                             int64_t(size * sizeof(int)));
    }
};

bool Fixture::no_check = false;

BENCHMARK_DEFINE_F(Fixture, BenchGPU)
(benchmark::State& st)
{
    this->bench_gpu<unsigned char>(st,
                                   radix_sort_basic<unsigned char>,
                                   256 * 1000000);
}

BENCHMARK_DEFINE_F(Fixture, BenchCPU)
(benchmark::State& st) { this->bench_cpu<unsigned char>(st, 256 * 1000000); }

BENCHMARK_REGISTER_F(Fixture, BenchGPU)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(Fixture, BenchCPU)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

int main(int argc, char** argv)
{
    ::benchmark::Initialize(&argc, argv);

    for (int i = 1; i < argc; i++)
    {
        if (argv[i] == std::string_view("--no-check"))
        {
            Fixture::no_check = true;
            std::swap(argv[i], argv[--argc]);
        }
    }

    ::benchmark::RunSpecifiedBenchmarks();
}

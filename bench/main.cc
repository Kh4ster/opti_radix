#include "cuda_tools/host_shared_ptr.cuh"
#include "test_helpers.hh"
#include "to_bench.cuh"

#include <benchmark/benchmark.h>

class Fixture : public benchmark::Fixture
{
  public:
    static bool no_check;

    template <typename FUNC, typename... Args>
    void
    bench(benchmark::State& st, FUNC callback, std::size_t size, Args&&... args)
    {
        constexpr int val = 5;
        cuda_tools::host_shared_ptr<int> buffer(size);

        for (auto _ : st)
        {
            buffer.fill(val);
            callback(buffer, std::forward<Args>(args)...);
        }

        st.SetBytesProcessed(int64_t(st.iterations()) *
                             int64_t(size * sizeof(int)));

        auto lambda = [val](int i) { return i == val + 1; };
        if (!no_check)
            check_buffer(buffer, lambda);
    }
};

bool Fixture::no_check = false;

// Basic bench
// Remove me (it is simply a sample)
BENCHMARK_DEFINE_F(Fixture, First_Bench)
(benchmark::State& st)
{
    this->bench(st, to_bench_single, std::size_t(1) << 25);
}

BENCHMARK_REGISTER_F(Fixture, First_Bench)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// Bench a function with multiple arguments
// Remove me (it is simply a sample)
BENCHMARK_DEFINE_F(Fixture, Bench_Multiple_Args)
(benchmark::State& st)
{
    this->bench(st, to_bench_multiple, std::size_t(1) << 25, 64, 1);
}

BENCHMARK_REGISTER_F(Fixture, Bench_Multiple_Args)
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

#include "cuda_tools/host_shared_ptr.cuh"
#include "test_helpers.hh"
#include "to_bench.cuh"

#include <benchmark/benchmark.h>

class Fixture : public benchmark::Fixture
{
  public:
    static bool no_check;

    template <typename... Args>
    void bench(benchmark::State& st,
               void (*callback)(cuda_tools::host_shared_ptr<int>, Args...),
               std::size_t size,
               Args... args)
    {
        cuda_tools::host_shared_ptr<int> buffer(size);

        for (auto _ : st)
            callback(buffer, args...);

        st.SetBytesProcessed(int64_t(st.iterations()) *
                             int64_t(size * sizeof(int)));

        if (!no_check)
            check_buffer(buffer);
    }
};

bool Fixture::no_check = false;

BENCHMARK_DEFINE_F(Fixture, First_Bench)
(benchmark::State& st) { this->bench(st, to_bench, 1 << 9); }

BENCHMARK_REGISTER_F(Fixture, First_Bench)
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

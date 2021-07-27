# Benchmark template project

The goal of this project is to be forked to serve as a base for anyone needing to benchmark several Cuda functions quickly.

## Requirements

* [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
* C++ compiler ([g++](https://gcc.gnu.org/) for linux,  [MSVC](https://visualstudio.microsoft.com/downloads/) for Windows)
* [GPU supported by CUDA](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
* [CMake](https://cmake.org/download/)
* [Conan](https://conan.io/center/)

### Additional libraries

These libraries are included in the conan file. Do not install them yourself. Conan will do the job for you.

* [GoogleTest](https://github.com/google/googletest)
* [GoogleBenchmark](https://github.com/google/benchmark)

## Build

- To build, execute the following commands :

```bash
mkdir build && cd build
conan install ..
cmake ..
make
```

## Run :

```bash
cd build
./bin/Bench
```

- Additionnaly you can use `--build=missing` to build missing libraries:

```bash
conan install .. --build missing
```

---

### Additional infos

* By default the program **will run in release** when it's inside a `build` or `build_release` folder. To build in **debug**, build the projet inside a `build_debug` folder.

* You can specify the "--no-check" option when running the bench binary to disable result checking :
```bash
./bin/Bench --no-check
```

## To bench your own code

### Create a new cmake target

In `CMakeLists.txt`:
* Create a new library as follows:
```cmake
add_library(LIB_NAME
	SOURCE_FILE1
	SOURCE_FILE2
	...
)
```
* Link this library to the `Bench` target (add this library among the others) as follows:
```cmake
target_link_libraries(Bench LIB_NAME async_memcpy GTest::GTest benchmark::benchmark TestHelpers)
```
* Note: be careful not to have the same functions name for exported functions between libraries (avoid multiple definition compilation error)

### Create new benches

In `bench/main.cc`:
* Include the header file containing the function(s) to bench `#include header_file.cuh` if not already in the included files
* Define a new bench as follows
```c++
BENCHMARK_DEFINE_F(Fixture, BENCH_NAME)
(benchmark::State &st)
{
    this->bench(st, NAME_OF_THE_FUNCTION_TO_BENCH, BUFFER_SIZE);
}
```
* Register the new bench as follows
```c++
BENCHMARK_REGISTER_F(Fixture, BENCH_NAME)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
```
* Note: a template of this steps can directly be found in `src/main.cc`

### Extra ressources
- You can use premade host_shared_ptr to allocate data
- You can use premade test_helper to test your result
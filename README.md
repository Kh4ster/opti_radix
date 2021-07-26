# Benchmark template projet

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

- Additionnaly you can use "--build missing" to build missing libraries:

```bash
conan install .. --build missing
```

---

### Additional infos

By default the program **will run in release** when it's inside a build / build_release folder. To build in debug, build the projet inside a build_debug folder.

You can specify the "--no-check" option when running the bench binary to disable result checking :
```bash
./bin/Bench --no-check
```

## To bench your own code

- Place the functions you need to bench inside "src/to_bench.cu"
- Add them in "bench/main.cc" to enable benching
- You can use premade host_shared_ptr to allocate data
- You can use premade test_helper to test your result

## Prerequisited
- conan
- cmake
- nvcc
- gcc/g++
- gtest
- google_benchmark
#include <chrono>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#define ITEMS_COUNT 1024 * 1024 * 256 * 4 / 2

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  }


// Generate random data on CPU and send it to GPU with id gpu_id
void generate_data(int gpu_id, int *host_buffer, int *gpu_buffer,
    size_t data_size, cudaStream_t stream) {
    // Generate random data on CPU
    cudaEvent_t *timing_events;
    for (int j = 0; j < data_size / sizeof(int); j++) {
    host_buffer[j] = 1;
    }
    // Transfer data to GPU
    CHECK_CUDA(cudaSetDevice(gpu_id));
    CHECK_CUDA(cudaMemcpyAsync(gpu_buffer, host_buffer, data_size,
                cudaMemcpyHostToDevice));
}
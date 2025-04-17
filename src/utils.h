#include <cooperative_groups.h>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <random>

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
// void generate_data(int gpu_id, int *host_buffer, int *gpu_buffer,
//                    size_t data_size) {
//   // Generate random data on CPU
//   cudaEvent_t *timing_events;
//   for (int j = 0; j < data_size / sizeof(int); j++) {
//     host_buffer[j] = 1;
//   }
//   // Transfer data to GPU
//   CHECK_CUDA(cudaSetDevice(gpu_id));
//   CHECK_CUDA(
//       cudaMemcpy(gpu_buffer, host_buffer, data_size,
//       cudaMemcpyHostToDevice));
// }

void generate_data(int gpu_id, int *gpu_buffer, size_t tuples_count,
                   int offset = 0, float randomness_factor) {
  // Generate random data on CPU
  // simple boundary checks:
  if (randomness_factor < 0.0f) randomness_factor = 0.0f;
  if (randomness_factor > 1.0f) randomness_factor = 1.0f;

  // WARN: number of potential unique values depends on tuples_count
  size_t unique_values = std::max((size_t)(tuples_count * randomness_factor), (size_t)1);


  int *host_buffer = (int *)malloc(tuples_count * sizeof(int));

  std::default_random_engine rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, unique_values - 1);

  int val = offset + dist(rng);

  for (int j = 0; j < tuples_count; j++) {
    if (randomness_factor == 0.0f) {
      val = offset + j;
    }
    host_buffer[j] = val;
  }
  for (int j = 0; j < 10; j++) { // Print first 10 values
    std::cout << host_buffer[j] << " ";
  }
  std::cout << std::endl;
    // Transfer data to GPU
  CHECK_CUDA(cudaSetDevice(gpu_id));
  CHECK_CUDA(cudaMemcpy(gpu_buffer, host_buffer, tuples_count * sizeof(int),
                        cudaMemcpyHostToDevice));
  free(host_buffer);
}


void generate_data(int gpu_id, int *gpu_buffer, size_t tuples_count,
                   int offset = 0) {
  // Generate random data on CPU
  int *host_buffer = (int *)malloc(tuples_count * sizeof(int));
  cudaEvent_t *timing_events;
  for (int j = 0; j < tuples_count; j++) {
    host_buffer[j] = offset + j;
  }
    // Transfer data to GPU
  CHECK_CUDA(cudaSetDevice(gpu_id));
  CHECK_CUDA(cudaMemcpy(gpu_buffer, host_buffer, tuples_count * sizeof(int),
                        cudaMemcpyHostToDevice));
  free(host_buffer);
}
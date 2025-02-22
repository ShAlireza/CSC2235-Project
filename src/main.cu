#include <chrono>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define DATA_SIZE 1024 * 1024 * 256
#define SRC_GPU 0
#define DEST_GPU 1

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  }

namespace cg = cooperative_groups;

__device__ void reduceBlock(long *sdata, const cg::thread_block &cta) {
  const unsigned int tid = cta.thread_rank();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  sdata[tid] = cg::reduce(tile32, sdata[tid], cg::plus<long>());
  cg::sync(cta);

  long beta = 0;
  if (cta.thread_rank() == 0) {
    beta = 0;
    for (int i = 0; i < blockDim.x; i += tile32.size()) {
      beta += sdata[i];
    }
    sdata[0] = beta;
  }
  cg::sync(cta);
}

extern "C" __global__ void reduceSinglePassMultiBlockCG(const int *g_idata,
                                                        long *g_odata, long n) {
  // Handle to thread block group
  cg::thread_block block = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();

  extern long __shared__ sdata[];

  sdata[block.thread_rank()] = 0;

  for (long i = grid.thread_rank(); i < n; i += grid.size()) {
    sdata[block.thread_rank()] += g_idata[i];
  }

  cg::sync(block);

  reduceBlock(sdata, block);

  if (block.thread_rank() == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

  cg::sync(grid);

  if (grid.thread_rank() == 0) {
    for (int block = 1; block < gridDim.x; block++) {
      g_odata[0] += g_odata[block];
    }
  }
}

void getNumBlocksAndThreads(long n, int maxBlocks, int maxThreads, int &blocks,
                            int &threads) {
  if (n == 1) {
    threads = 1;
    blocks = 1;
  } else {
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &blocks, &threads, reduceSinglePassMultiBlockCG));
  }

  if (maxBlocks < blocks) {
    blocks = maxBlocks;
  }
}

void run_cuda_sum(int device, int *data, cudaEvent_t **timing_events, cudaStream_t stream) {
  CHECK_CUDA(cudaSetDevice(device));
  long size;

  // Set the device to be used
  cudaDeviceProp prop = {0};
  CHECK_CUDA(cudaSetDevice(device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  size = DATA_SIZE;

  // Determine the launch configuration (threads, blocks)
  int maxThreads = 0;
  int maxBlocks = 0;

  maxThreads = prop.maxThreadsPerBlock;

  maxBlocks = prop.multiProcessorCount *
              (prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);

  int numBlocks = 0;
  int numThreads = 0;
  getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

  // We calculate the occupancy to know how many block can actually fit on the
  // GPU
  int numBlocksPerSm = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, reduceSinglePassMultiBlockCG, numThreads,
      numThreads * sizeof(long)));

  int numSms = prop.multiProcessorCount;

  if (numBlocks > numBlocksPerSm * numSms) {
    numBlocks = numBlocksPerSm * numSms;
  }

  int *result;
  CHECK_CUDA(cudaMalloc(&result, sizeof(int)));

  int smemSize = numThreads * sizeof(long);
  void *kernelArgs[] = {
      (void *)&data,
      (void *)&result,
      (void *)&size,
  };

  dim3 dimBlock(numThreads, 1, 1);
  dim3 dimGrid(numBlocks, 1, 1);
  cudaEvent_t *events = (cudaEvent_t *)malloc(2 * sizeof(cudaEvent_t));
  cudaEventCreate(&events[0]);
  cudaEventCreate(&events[1]);

  CHECK_CUDA(cudaEventRecord(events[0], stream));
  cudaLaunchCooperativeKernel((void *)reduceSinglePassMultiBlockCG, dimGrid,
                              dimBlock, kernelArgs, smemSize, stream);
  CHECK_CUDA(cudaEventRecord(events[1], stream));

  int validate_result;
  CHECK_CUDA(cudaMemcpy(&validate_result, result, sizeof(int),
                        cudaMemcpyDeviceToHost));

  *timing_events = events;
  printf("Final result from GPU: %d\n", validate_result);
}

long openmp_sum(int *data, size_t size) {
  long sum = 0;
  int num_threads = omp_get_max_threads();
#pragma omp parallel for simd reduction(+ : sum) num_threads(num_threads)
  for (size_t i = 0; i < size; i++) {
    sum += data[i];
  }

  return sum;
}

// Transfer data host-to-device or device-to-host async
void transfer_data(int gpu_id, int *src_data, int *host_buffer,
                   size_t data_size, cudaStream_t stream,
                   cudaEvent_t **timing_events, bool dtoh = true) {
  CHECK_CUDA(cudaSetDevice(gpu_id));

  cudaEvent_t *events = (cudaEvent_t *)malloc(2 * sizeof(cudaEvent_t));

  CHECK_CUDA(cudaEventCreate(&events[0]));
  CHECK_CUDA(cudaEventCreate(&events[1]));

  CHECK_CUDA(cudaEventRecord(events[0], stream));
  if (dtoh) {
    CHECK_CUDA(cudaMemcpyAsync(host_buffer, src_data, data_size,
                               cudaMemcpyDeviceToHost, stream));
  } else {
    CHECK_CUDA(cudaMemcpyAsync(src_data, host_buffer, data_size,
                               cudaMemcpyHostToDevice, stream));
  }
  CHECK_CUDA(cudaEventRecord(events[1], stream));
  *timing_events = events;
}

// TODO: Validate data on destination

// Generate random data on CPU and send it to GPU with id i
void generate_data(int gpu_id, int *host_buffer, int *gpu_buffer,
                   size_t data_size, cudaStream_t stream) {
  // Generate random data on CPU
  //
  cudaEvent_t *timing_events;
  for (int j = 0; j < data_size / sizeof(int); j++) {
    host_buffer[j] = rand();
  }
  // Transfer data to GPU
  transfer_data(gpu_id, gpu_buffer, host_buffer, data_size, stream,
                &timing_events, false);
}

void compute_on_destination(int src_gpu, int dest_gpu, int *host_buffer,
                            int *src_data, int *dest_data) {
  memset(host_buffer, 0, DATA_SIZE * sizeof(int));

  cudaStream_t src_stream, dest_stream;

  CHECK_CUDA(cudaSetDevice(SRC_GPU));
  CHECK_CUDA(cudaStreamCreate(&src_stream));
  cudaEvent_t *timing_events_src_host;
  transfer_data(SRC_GPU, src_data, host_buffer, DATA_SIZE * sizeof(int),
                src_stream, &timing_events_src_host);

  CHECK_CUDA(cudaSetDevice(DEST_GPU));
  CHECK_CUDA(cudaStreamCreate(&dest_stream));

  cudaEvent_t *timing_events_host_dest;
  transfer_data(DEST_GPU, dest_data, host_buffer, DATA_SIZE * sizeof(int),
                dest_stream, &timing_events_host_dest, false);

  CHECK_CUDA(cudaSetDevice(SRC_GPU));
  CHECK_CUDA(cudaStreamSynchronize(src_stream));
  CHECK_CUDA(cudaSetDevice(DEST_GPU));
  CHECK_CUDA(cudaStreamSynchronize(dest_stream));

  float src_host_timing, host_dest_timing;

  CHECK_CUDA(cudaEventElapsedTime(&src_host_timing, timing_events_src_host[0],
                                  timing_events_src_host[1]));
  CHECK_CUDA(cudaEventElapsedTime(&host_dest_timing, timing_events_host_dest[0],
                                  timing_events_host_dest[1]));

  printf("Src to Host: %f\n", src_host_timing);
  printf("Host to Dest: %f\n", host_dest_timing);
  cudaEvent_t *sum_reduction_events;
  run_cuda_sum(DEST_GPU, dest_data, &sum_reduction_events, dest_stream);

  float reduction_time;
  CHECK_CUDA(cudaEventElapsedTime(&reduction_time, sum_reduction_events[0],
                                  sum_reduction_events[1]));

  printf("Reduction time on GPU: %f\n", reduction_time);
}

void compute_on_path(int src_gpu, int dest_gpu, int *host_buffer, int *src_data,
                     int *dest_data) {
  cudaStream_t src_stream, dest_stream;

  CHECK_CUDA(cudaSetDevice(SRC_GPU));
  CHECK_CUDA(cudaStreamCreate(&src_stream));

  memset(host_buffer, 0, DATA_SIZE * sizeof(int));

  cudaEvent_t *timing_events_src_host;
  transfer_data(SRC_GPU, src_data, host_buffer, DATA_SIZE * sizeof(int),
                src_stream, &timing_events_src_host);

  auto start = std::chrono::high_resolution_clock::now();
  int result = openmp_sum(host_buffer, DATA_SIZE);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  printf("Timing for OpenMP sum: %ld\n", duration.count());

  CHECK_CUDA(cudaSetDevice(DEST_GPU));
  CHECK_CUDA(cudaStreamCreate(&dest_stream));

  cudaEvent_t *timing_events_host_dest;
  transfer_data(DEST_GPU, dest_data, &result, sizeof(int), dest_stream,
                &timing_events_host_dest, false);

  CHECK_CUDA(cudaSetDevice(SRC_GPU));
  CHECK_CUDA(cudaStreamSynchronize(src_stream));
  CHECK_CUDA(cudaSetDevice(DEST_GPU));
  CHECK_CUDA(cudaStreamSynchronize(dest_stream));

  float src_host_timing, host_dest_timing;

  CHECK_CUDA(cudaEventElapsedTime(&src_host_timing, timing_events_src_host[0],
                                  timing_events_src_host[1]));
  CHECK_CUDA(cudaEventElapsedTime(&host_dest_timing, timing_events_host_dest[0],
                                  timing_events_host_dest[1]));

  printf("Src to Host: %f\n", src_host_timing);
  printf("Host to Dest: %f\n", host_dest_timing);

  CHECK_CUDA(cudaSetDevice(DEST_GPU));
  int validate_result;
  CHECK_CUDA(cudaMemcpy(&validate_result, dest_data, sizeof(int),
                        cudaMemcpyDeviceToHost));
  printf("Final result from GPU: %d\n", validate_result);
  printf("Final result from Intermediate HOST: %d\n", result);
}

int main(int argc, char **argv) {
  CHECK_CUDA(cudaSetDevice(SRC_GPU));
  int *host_buffer = (int *)malloc(DATA_SIZE * sizeof(int));
  int *src_gpu_data;
  CHECK_CUDA(cudaMalloc((void **)&src_gpu_data, DATA_SIZE * sizeof(int)));

  CHECK_CUDA(cudaSetDevice(DEST_GPU));
  int *dest_gpu_data;
  CHECK_CUDA(cudaMalloc((void **)&dest_gpu_data, DATA_SIZE * sizeof(int)));

  generate_data(0, host_buffer, src_gpu_data, DATA_SIZE * sizeof(int), 0);

  compute_on_destination(SRC_GPU, DEST_GPU, host_buffer, src_gpu_data,
                         dest_gpu_data);
  compute_on_path(SRC_GPU, DEST_GPU, host_buffer, src_gpu_data, dest_gpu_data);

  return 0;
}

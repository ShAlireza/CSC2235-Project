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

#define DATA_SIZE 16
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

// Sum reduction GPU kernels
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

// finding the optimzal number of threads and blocks for the given kernel
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

void run_cuda_sum(int device, int *data, cudaEvent_t **timing_events,
                  cudaStream_t stream, int chunk_size) {
  CHECK_CUDA(cudaSetDevice(device));
  long size;

  // Set the device to be used
  cudaDeviceProp prop = {0};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  size = chunk_size;

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

  // Allocate memory for the result on the device
  int *result;
  CHECK_CUDA(cudaMallocAsync(&result, sizeof(int), stream));

  // Inputs to the sum reduction kernel
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

  // Validate the results on GPU
  int validate_result;
  CHECK_CUDA(cudaMemcpyAsync(&validate_result, result, sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  *timing_events = events;
  printf("Final result from GPU(compute on destination): %d\n",
         validate_result);
}

// OpenMP implementation of sum reduction using all available cpu cores
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
                   size_t data_size, cudaEvent_t **timing_events,
                   bool dtoh = true) {
  CHECK_CUDA(cudaSetDevice(gpu_id));

  cudaEvent_t *events = (cudaEvent_t *)malloc(2 * sizeof(cudaEvent_t));

  CHECK_CUDA(cudaEventCreate(&events[0]));
  CHECK_CUDA(cudaEventCreate(&events[1]));

  CHECK_CUDA(cudaEventRecord(events[0]));
  if (dtoh) {
    CHECK_CUDA(
        cudaMemcpy(host_buffer, src_data, data_size, cudaMemcpyDeviceToHost));
  } else {
    CHECK_CUDA(
        cudaMemcpy(src_data, host_buffer, data_size, cudaMemcpyHostToDevice));
  }
  CHECK_CUDA(cudaEventRecord(events[1]));
  *timing_events = events;
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
  transfer_data(gpu_id, gpu_buffer, host_buffer, data_size, &timing_events,
                false);
}

// Baseline implementation of sum reduction on destination GPU without any
// pipelining
void compute_on_destination(int src_gpu, int dest_gpu, int *host_buffer,
                            int *src_data, int *dest_data) {
  printf("Starting compute on destination Scenario\n");
  // resetting the host buffer
  memset(host_buffer, 0, DATA_SIZE * sizeof(int));

  cudaStream_t src_stream, dest_stream;

  CHECK_CUDA(cudaSetDevice(SRC_GPU));
  CHECK_CUDA(cudaStreamCreate(&src_stream));
  cudaEvent_t *timing_events_src_host;
  // Transfer data from SRC to HOST buffer
  transfer_data(SRC_GPU, src_data, host_buffer, DATA_SIZE * sizeof(int),
                &timing_events_src_host);

  CHECK_CUDA(cudaSetDevice(DEST_GPU));
  CHECK_CUDA(cudaStreamCreate(&dest_stream));

  cudaEvent_t *timing_events_host_dest;
  // Transfer data from HOST to DEST buffer
  transfer_data(DEST_GPU, dest_data, host_buffer, DATA_SIZE * sizeof(int),
                &timing_events_host_dest, false);

  CHECK_CUDA(cudaSetDevice(SRC_GPU));
  CHECK_CUDA(cudaStreamSynchronize(src_stream));

  // Do the final global sum on destination GPU
  cudaEvent_t *sum_reduction_events;
  int *sum_result;
  CHECK_CUDA(cudaMalloc((void **)&sum_result, sizeof(int)));
  run_cuda_sum(DEST_GPU, dest_data, &sum_reduction_events, 0, DATA_SIZE);

  CHECK_CUDA(cudaSetDevice(DEST_GPU));
  CHECK_CUDA(cudaStreamSynchronize(dest_stream));

  float reduction_time;
  CHECK_CUDA(cudaEventElapsedTime(&reduction_time, sum_reduction_events[0],
                                  sum_reduction_events[1]));

  printf("Reduction time on GPU: %f\n", reduction_time);

  float src_host_timing, host_dest_timing;

  CHECK_CUDA(cudaEventElapsedTime(&src_host_timing, timing_events_src_host[0],
                                  timing_events_src_host[1]));
  CHECK_CUDA(cudaEventElapsedTime(&host_dest_timing, timing_events_host_dest[0],
                                  timing_events_host_dest[1]));

  printf("Src to Host: %f\n", src_host_timing);
  printf("Host to Dest: %f\n", host_dest_timing);
}

void compute_on_path(int src_gpu, int dest_gpu, int *host_buffer, int *src_data,
                     int *dest_data) {
  printf("Starting compute on path Scenario\n");
  cudaStream_t src_stream, dest_stream;

  CHECK_CUDA(cudaSetDevice(SRC_GPU));
  CHECK_CUDA(cudaStreamCreate(&src_stream));

  // resetting the host buffer
  memset(host_buffer, 0, DATA_SIZE * sizeof(int));

  cudaEvent_t *timing_events_src_host;
  // Transfer data from SRC to HOST buffer
  transfer_data(SRC_GPU, src_data, host_buffer, DATA_SIZE * sizeof(int),
                &timing_events_src_host);

  // Do the sum reduction on the intermediate host using openmp
  auto start = std::chrono::high_resolution_clock::now();
  int result = openmp_sum(host_buffer, DATA_SIZE);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  printf("Timing for OpenMP sum: %ld\n", duration.count());

  CHECK_CUDA(cudaSetDevice(DEST_GPU));
  CHECK_CUDA(cudaStreamCreate(&dest_stream));

  cudaEvent_t *timing_events_host_dest;
  // Transfer data from HOST to DEST buffer
  transfer_data(DEST_GPU, dest_data, &result, sizeof(int),
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

void compute_on_destination_thread(int src_gpu, int dest_gpu, int *host_buffer,
                                   int *src_gpu_data, int *dest_gpu_data,
                                   int *sum_result, int start_index,
                                   int chunk_size, int thread_index) {

  cudaEvent_t *first_copy_events =
      (cudaEvent_t *)malloc(2 * sizeof(cudaEvent_t));
  cudaEvent_t *second_copy_events =
      (cudaEvent_t *)malloc(2 * sizeof(cudaEvent_t));
  CHECK_CUDA(cudaSetDevice(SRC_GPU));
  CHECK_CUDA(cudaEventCreate(&first_copy_events[0]));
  CHECK_CUDA(cudaEventCreate(&first_copy_events[1]));
  printf("[%d]: Starting thread\n", thread_index);
  printf("[%d]: Chunk size: %d\n", thread_index, chunk_size);
  printf("[%d]: Start index: %d\n", thread_index, start_index);
  // transfer_data(SRC_GPU, &src_gpu_data[start_index],
  // &host_buffer[start_index],
  //               chunk_size,  &first_copy_events);

  CHECK_CUDA(cudaEventRecord(first_copy_events[0]));
  CHECK_CUDA(cudaMemcpy(&host_buffer[start_index], &src_gpu_data[start_index],
                        chunk_size, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaEventRecord(first_copy_events[1]));

  // CHECK_CUDA(cudaStreamSynchronize(0));
  printf("[%d]: First copy done\n", thread_index);

  CHECK_CUDA(cudaSetDevice(DEST_GPU));
  CHECK_CUDA(cudaEventCreate(&second_copy_events[0]));
  CHECK_CUDA(cudaEventCreate(&second_copy_events[1]));
  // transfer_data(DEST_GPU, &dest_gpu_data[start_index],
  //               &host_buffer[start_index], chunk_size, &second_copy_events,
  //               false);
  CHECK_CUDA(cudaEventRecord(second_copy_events[0]));
  CHECK_CUDA(cudaMemcpy(&dest_gpu_data[start_index], &host_buffer[start_index],
                        chunk_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaEventRecord(second_copy_events[1]));
  // CHECK_CUDA(cudaStreamSynchronize(0));
  printf("[%d]: Second copy done\n", thread_index);

  cudaEvent_t *sum_reduction_events;
  run_cuda_sum(DEST_GPU, dest_gpu_data + start_index, &sum_reduction_events, 0,
               chunk_size / sizeof(int)); // TODO: Refactor chunk size

  float first_copy_time, second_copy_time, reduction_time;
  CHECK_CUDA(cudaEventElapsedTime(&first_copy_time, first_copy_events[0],
                                  first_copy_events[1]));
  CHECK_CUDA(cudaEventElapsedTime(&second_copy_time, second_copy_events[0],
                                  second_copy_events[1]));
  CHECK_CUDA(cudaEventElapsedTime(&reduction_time, sum_reduction_events[0],
                                  sum_reduction_events[1]));

  printf("[%d]: First copy time: %f\n", thread_index, first_copy_time);
  printf("[%d]: Second copy time: %f\n", thread_index, second_copy_time);
  printf("[%d]: Reduction time: %f\n", thread_index, reduction_time);
}

void compute_on_destination_pipelined(int src_gpu, int dest_gpu,
                                      int *host_buffer, int *src_gpu_data,
                                      int *dest_gpu_data, int threads_count) {

  int *sum_results;

  CHECK_CUDA(cudaSetDevice(DEST_GPU));
  CHECK_CUDA(cudaMalloc((void **)&sum_results, sizeof(int) * threads_count));

  std::thread threads[threads_count];
  int items_per_thread = DATA_SIZE / threads_count;

  for (int i = 0; i < threads_count; i++) {
    printf("Starting thread %d\n", i);
    printf("Chunk size: %ld\n", items_per_thread * sizeof(int));
    printf("Start index: %d\n", i * items_per_thread);
    threads[i] =
        std::thread(compute_on_destination_thread, src_gpu, dest_gpu,
                    host_buffer, src_gpu_data, dest_gpu_data, &sum_results[i],
                    i * items_per_thread, items_per_thread * sizeof(int), i);
  }

  for (int i = 0; i < threads_count; i++) {
    threads[i].join();
  }
}

int main(int argc, char **argv) {
  // Initialize the initial state of the cluster (src data and buffers)
  CHECK_CUDA(cudaSetDevice(SRC_GPU));
  int *host_buffer = (int *)malloc(DATA_SIZE * sizeof(int));
  int *src_gpu_data;
  CHECK_CUDA(cudaMalloc((void **)&src_gpu_data, DATA_SIZE * sizeof(int)));

  CHECK_CUDA(cudaSetDevice(DEST_GPU));
  int *dest_gpu_data;
  CHECK_CUDA(cudaMalloc((void **)&dest_gpu_data, DATA_SIZE * sizeof(int)));

  generate_data(0, host_buffer, src_gpu_data, DATA_SIZE * sizeof(int), 0);

  // Run the two scenarios

  // compute_on_destination(SRC_GPU, DEST_GPU, host_buffer, src_gpu_data,
  //                        dest_gpu_data);
  compute_on_destination_pipelined(SRC_GPU, DEST_GPU, host_buffer, src_gpu_data,
                                   dest_gpu_data, 4);
  // compute_on_path(SRC_GPU, DEST_GPU, host_buffer, src_gpu_data,
  // dest_gpu_data);

  cudaFree(src_gpu_data);
  cudaFree(dest_gpu_data);
  free(host_buffer);

  return 0;
}

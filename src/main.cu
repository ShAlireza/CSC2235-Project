#include <cstring>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#define DATA_SIZE (1 << 18)
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

// Transfer data host-to-device or device-to-host async
void transfer_data(int gpu_id, int *src_data, int *host_buffer,
                   size_t data_size, cudaStream_t stream,
                   cudaEvent_t *timing_events, bool dtoh = true) {
  CHECK_CUDA(cudaSetDevice(gpu_id));

  timing_events = (cudaEvent_t *)malloc(2 * sizeof(cudaEvent_t));

  CHECK_CUDA(cudaEventCreate(&timing_events[0]));
  CHECK_CUDA(cudaEventCreate(&timing_events[1]));

  CHECK_CUDA(cudaEventRecord(timing_events[0], stream));
  if (dtoh) {
    CHECK_CUDA(cudaMemcpyAsync(host_buffer, src_data, data_size,
                               cudaMemcpyDeviceToHost, stream));
  } else {
    CHECK_CUDA(cudaMemcpyAsync(src_data, host_buffer, data_size,
                               cudaMemcpyHostToDevice, stream));
  }
  CHECK_CUDA(cudaEventRecord(timing_events[1], stream));
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
                timing_events, false);
}

int main(int argc, char **argv) {
  CHECK_CUDA(cudaSetDevice(SRC_GPU));
  int *data = (int *)malloc(DATA_SIZE * sizeof(int));
  int *gpu_data;
  CHECK_CUDA(cudaMalloc((void **)&gpu_data, DATA_SIZE * sizeof(int)));

  cudaStream_t stream;

  CHECK_CUDA(cudaStreamCreate(&stream));

  generate_data(0, data, gpu_data, DATA_SIZE * sizeof(int), stream);

  memset(data, 0, DATA_SIZE * sizeof(int));

  cudaEvent_t *timing_events_src_host;
  transfer_data(SRC_GPU, gpu_data, data, DATA_SIZE * sizeof(int), stream, timing_events_src_host);

  cudaEvent_t *timing_events_host_dest;
  transfer_data(DEST_GPU, gpu_data, data, DATA_SIZE * sizeof(int), stream, timing_events_host_dest, false);

  CHECK_CUDA(cudaStreamSynchronize(stream));

  float src_host_timing, host_dest_timing;

  CHECK_CUDA(cudaEventElapsedTime(&src_host_timing, timing_events_src_host[0], timing_events_src_host[1]));
  CHECK_CUDA(cudaEventElapsedTime(&host_dest_timing, timing_events_host_dest[0], timing_events_host_dest[1]));


  printf("Src to Host: %f\n", src_host_timing);
  printf("Host to Dest: %f\n", host_dest_timing);

  return 0; 
}

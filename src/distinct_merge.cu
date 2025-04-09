#include "distinct_merge.h"
#include "utils.h"
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <math.h>
#include <mutex>
#include <thread>

DistinctMerge::DistinctMerge(const std::vector<int *> receive_buffers,
                             const std::vector<int> receive_buffer_sizes)
    : receive_buffers(receive_buffers),
      receive_buffer_sizes(receive_buffer_sizes) {

  // print the receive buffers
  // std::cout << "Receive buffers: " << std::endl;
  // for (int i = 0; i < receive_buffers.size(); i++) {
  //   std::cout << "Buffer " << i << ": ";
  //   for (int j = 0; j < receive_buffer_sizes[i]; j++) {
  //     std::cout << receive_buffers[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  //std::thread sender_thread(&DistinctMerge::sender, this);
}

int DistinctMerge::check_value(int value) {
  // WARN: We should remove locking later since its a performance bottleneck (we
  // should use somthing like Intel TBB)

  std::unique_lock<std::mutex> lock(this->seen_values_mutex);

  auto it = seen_values.find(value);
  if (it != seen_values.end()) {
    // INFO: We assume that input data are positive integers
    lock.unlock();
    return -1;
  } else {
    seen_values[value] = true;
    lock.unlock();
    return value;
  }
}

bool DistinctMerge::stage(int value) {

  std::unique_lock<std::mutex> lock(this->send_buffer_mutex);

  this->send_buffer[this->send_buffer_end_index++] = value;

  lock.unlock();

  return true;
}

void DistinctMerge::sender() {
  // TODO: this function check the send buffer and sends data whenever it
  // reached the threshold

  while (true) {
    int difference =
        std::abs(this->send_buffer_start_index - this->send_buffer_end_index);
    if (difference >= DISTINCT_MERGE_BUFFER_THRESHOLD) {
      std::cout << "Sending data" << std::endl;
      this->send_buffer_start_index += difference;
    }
  }
}

DistinctMergeGPU::DistinctMergeGPU(int gpu_id, int tuples_count, int chunk_size)
    : gpu_id(gpu_id), tuples_count(tuples_count), chunk_size(chunk_size) {
  // TODO: init random data on gpu
  CHECK_CUDA(cudaSetDevice(gpu_id));
  CHECK_CUDA(cudaMalloc((void **)&this->gpu_data, tuples_count * sizeof(int)));
  generate_data(gpu_id, this->gpu_data, tuples_count);

  // TODO: allocate destination buffer on cpu
  this->destination_buffer = new int[tuples_count];
}

void DistinctMergeGPU::exec(int start_index) {
  // TODO: Run the deduplication on the chunk (do it later, for now we just
  // assume that all tuples have unique values)

  // TODO: Send the deduplicated chunk to CPU
  CHECK_CUDA(cudaMemcpy(
      this->destination_buffer + start_index, this->gpu_data + start_index,
      this->chunk_size * sizeof(int), cudaMemcpyDeviceToHost));

  // TODO: Check the values and stage them for sending
  for (int i = start_index; i < start_index + this->chunk_size; i++) {
    int checked_value =
        this->cpu_merger->check_value(this->destination_buffer[i]);

    // Tuple is new so we should stage it into the send buffer
    if (checked_value != -1)
      this->cpu_merger->stage(checked_value);
  }
}

void DistinctMergeGPU::start() {
  std::cout << std::unitbuf;
  std::cout << "Starting GPU merger" << std::endl;

  int threads_count = this->tuples_count / this->chunk_size;

  std::thread threads[threads_count];

  for (int i = 0; i < threads_count; i++) {
    threads[i] =
        std::thread(&DistinctMergeGPU::exec, this, i * this->chunk_size);
  }

  for (int i = 0; i < threads_count; i++) {
    threads[i].join();
  }
}

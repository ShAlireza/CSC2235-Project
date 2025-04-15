#include "RDMA/ucx_rdma_client.h"
#include "distinct_merge.h"
#include "utils.h"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <mutex>
#include <thread>

DistinctMerge::DistinctMerge(const std::vector<int *> &receive_buffers,
                             const std::vector<int> &receive_buffer_sizes)
    : receive_buffers(receive_buffers),
      receive_buffer_sizes(receive_buffer_sizes) {

  this->send_buffer = (int *)malloc(DISTINCT_MERGE_BUFFER_SIZE * sizeof(int));

  this->sender_thread = std::thread(&DistinctMerge::sender, this);
  this->sender_thread.detach();
}

void DistinctMerge::set_rdma_client(UcxRdmaClient *client) {
  this->rdma_client = client;
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
    seen_values.emplace(value, true);
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

  std::cout << "In sender thread" << std::endl;
  while (true) {

    int difference =
        std::abs(this->send_buffer_start_index - this->send_buffer_end_index);

    if (difference >= DISTINCT_MERGE_BUFFER_THRESHOLD) {
      // std::cout << "[Sender] Threshold reached: " << difference << " values
      // ready\n";

      while (difference >= DISTINCT_MERGE_SEND_CHUNK_SIZE) {
        std::unique_lock<std::mutex> lock(this->send_buffer_mutex);
        int *chunk_ptr = &this->send_buffer[this->send_buffer_start_index];
        int chunk_bytes = DISTINCT_MERGE_SEND_CHUNK_SIZE * sizeof(int);
        lock.unlock();

        if (this->rdma_client != nullptr) {
          std::cout << "[Sender] Sending chunk of size " << chunk_bytes
                    << std::endl;

          this->rdma_client->send_chunk(chunk_ptr, chunk_bytes);
        } else {
          std::cout << "[Sender] RDMA client is not set, skipping send"
                    << std::endl;
        }
        this->send_buffer_start_index += DISTINCT_MERGE_SEND_CHUNK_SIZE;

        difference = std::abs(this->send_buffer_start_index -
                              this->send_buffer_end_index);
      }
    }

    if (this->finished) {
      std::cout << "Sender thread finished" << std::endl;

      if (difference > 0) {
        std::unique_lock<std::mutex> lock(this->send_buffer_mutex);
        int *chunk_ptr = &this->send_buffer[this->send_buffer_start_index];
        int chunk_bytes = difference * sizeof(int);
        lock.unlock();

        if (this->rdma_client != nullptr) {
          std::cout << "[Sender] Sending chunk of size " << chunk_bytes
                    << std::endl;

          this->rdma_client->send_chunk(chunk_ptr, chunk_bytes);
        } else {
          std::cout << "[Sender] RDMA client is not set, skipping send"
                    << std::endl;
        }
      }
      this->rdma_client->send_finish();
      std::cout << "Sender thread finished sending counter=-1" << std::endl;

      this->rdma_client->finish();
      while (!this->rdma_client->done_flushing);

      this->done_flushing = true;
      break;
    }
  }
}

void DistinctMerge::finish() { this->finished = true; }

DistinctMergeGPU::DistinctMergeGPU(int gpu_id, int tuples_count, int chunk_size)
    : gpu_id(gpu_id), tuples_count(tuples_count), chunk_size(chunk_size) {
  // TODO: init random data on gpu
  CHECK_CUDA(cudaSetDevice(gpu_id));
  CHECK_CUDA(cudaMalloc((void **)&this->gpu_data, tuples_count * sizeof(int)));
  generate_data(gpu_id, this->gpu_data, tuples_count, tuples_count * gpu_id);

  // TODO: allocate destination buffer on cpu
  // this->destination_buffer = new int[tuples_count];
  this->destination_buffer = (int *)malloc(tuples_count * sizeof(int));
}

void DistinctMergeGPU::exec(int start_index) {
  cudaSetDevice(this->gpu_id);
  // TODO: Run the deduplication on the chunk (do it later, for now we just
  // assume that all tuples have unique values)

  // TODO: Send the deduplicated chunk to CPU
  CHECK_CUDA(cudaMemcpy(
      this->destination_buffer + start_index, this->gpu_data + start_index,
      this->chunk_size * sizeof(int), cudaMemcpyDeviceToHost));

  // TODO: Check the values and stage them for sending
  int number_of_inserts = 0;
  for (int i = start_index; i < start_index + this->chunk_size; i++) {
    int checked_value =
        this->cpu_merger->check_value(this->destination_buffer[i]);

    // Tuple is new so we should stage it into the send buffer
    if (checked_value != -1) {
      number_of_inserts++;
      this->cpu_merger->stage(checked_value);
    }
  }
  // std::cout << "GPU: " << this->gpu_id
  //           << " - Number of inserts: " << number_of_inserts
  //           << " for chunk starting at index: " << start_index
  //           << "chunk size: " << this->chunk_size << std::endl;
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

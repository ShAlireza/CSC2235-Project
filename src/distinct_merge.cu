#include "RDMA/ucx_rdma_client.h"
#include "distinct_merge.h"
#include "utils.h"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <mutex>
#include <thread>

DistinctMerge::DistinctMerge(
    const std::vector<int *> &receive_buffers,
    const std::vector<unsigned long> &receive_buffer_sizes,
    unsigned long send_buffer_size, unsigned long send_buffer_threshold,
    TimeKeeper *timekeeper)
    : receive_buffers(receive_buffers),
      receive_buffer_sizes(receive_buffer_sizes),
      send_buffer_threshold(send_buffer_threshold), timekeeper(timekeeper) {

  this->send_buffer = (int *)malloc(send_buffer_size * sizeof(int));

  this->sender_thread = std::thread(&DistinctMerge::sender, this);
  this->sender_thread.detach();
}

void DistinctMerge::set_rdma_client(UcxRdmaClient *client) {
  this->rdma_client = client;
}

int DistinctMerge::check_value(int value) {
  // WARN: We should remove locking later since its a performance bottleneck (we
  // should use somthing like Intel TBB)
  // std::unique_lock<std::mutex> lock(this->seen_values_mutex);
  //
  // auto it = seen_values.find(value);
  // if (it != seen_values.end()) {
  //   // INFO: We assume that input data are positive integers
  //   lock.unlock();
  //   return -1;
  // } else {
  //   seen_values.emplace(value, true);
  //   lock.unlock();
  //   return value;
  // }
  return this->seen_values_concurrent.insert(value, true);
}

bool DistinctMerge::stage(int value) {

  std::unique_lock<std::mutex> lock(this->send_buffer_mutex);

  this->send_buffer[this->send_buffer_end_index++] = value;

  lock.unlock();

  return true;
}

bool DistinctMerge::stage_buffer(int *buffer, int tuples_count) {
  std::unique_lock<std::mutex> lock(this->send_buffer_mutex);

  memcpy(&this->send_buffer[this->send_buffer_end_index], buffer,
         tuples_count * sizeof(int));

  this->send_buffer_end_index += tuples_count;

  lock.unlock();
  return true;
}

void DistinctMerge::sender() {
  // TODO: this function check the send buffer and sends data whenever it
  // reached the threshold

  // std::cout << "In sender thread" << std::endl;
  while (true) {

    int difference =
        std::abs(this->send_buffer_start_index - this->send_buffer_end_index);

    // if (difference >= DISTINCT_MERGE_BUFFER_THRESHOLD) {
    // std::cout << "[Sender] Threshold reached: " << difference << " values
    // ready\n";

    while (difference >= this->send_buffer_threshold) {
      std::unique_lock<std::mutex> lock(this->send_buffer_mutex);
      int *chunk_ptr = &this->send_buffer[this->send_buffer_start_index];
      int chunk_bytes = this->send_buffer_threshold * sizeof(int);
      lock.unlock();

      if (this->rdma_client != nullptr) {
        // std::cout << "[Sender] Sending chunk of size " << chunk_bytes
        //           << std::endl;

        this->rdma_client->send_chunk(chunk_ptr, chunk_bytes);
      } else {
        std::cout << "[Sender] RDMA client is not set, skipping send"
                  << std::endl;
      }
      this->send_buffer_start_index += this->send_buffer_threshold;

      difference =
          std::abs(this->send_buffer_start_index - this->send_buffer_end_index);
    }
    // }

    if (this->finished) {
      // std::cout << "Sender thread finished" << std::endl;

      if (difference > 0) {
        std::unique_lock<std::mutex> lock(this->send_buffer_mutex);
        int *chunk_ptr = &this->send_buffer[this->send_buffer_start_index];
        int chunk_bytes = difference * sizeof(int);
        lock.unlock();

        if (this->rdma_client != nullptr) {
          // std::cout << "[Sender] Sending chunk of size " << chunk_bytes
          //           << std::endl;

          this->rdma_client->send_chunk(chunk_ptr, chunk_bytes);
        } else {
          std::cout << "[Sender] RDMA client is not set, skipping send"
                    << std::endl;
        }
      }
      this->rdma_client->send_finish();
      // std::cout << "Sender thread finished sending counter=-1" << std::endl;

      this->rdma_client->finish();
      while (!this->rdma_client->done_flushing)
        ;

      // std::cout << "Rdma client closed" << std::endl;
      this->done_flushing = true;
      break;
    }
  }
}

void DistinctMerge::finish() { this->finished = true; }

DistinctMergeGPU::DistinctMergeGPU(int gpu_id, int tuples_count, int chunk_size,
                                   bool deduplicate, TimeKeeper *timekeeper,
                                   float randomness)
    : gpu_id(gpu_id), tuples_count(tuples_count), chunk_size(chunk_size),
      deduplicate(deduplicate), timekeeper(timekeeper), randomness(randomness) {
  // TODO: init random data on gpu
  CHECK_CUDA(cudaSetDevice(gpu_id));
  CHECK_CUDA(cudaMalloc((void **)&this->gpu_data, tuples_count * sizeof(int)));
  // WARN: generated data shouldn't include 0 (always > 0)
  generate_data(gpu_id, this->gpu_data, tuples_count, this->randomness,
                tuples_count * gpu_id + 1);

  CHECK_CUDA(cudaMallocHost((void **)&this->destination_buffer,
                            tuples_count * sizeof(int)));
}

void DistinctMergeGPU::exec(int start_index) {
  cudaSetDevice(this->gpu_id);
  // TODO: Run the deduplication on the chunk (do it later, for now we just
  // assume that all tuples have unique values)

  // if (!this->first_chunk_started) {
  //   this->first_chunk_started = true;
  // print timestamp in nanoseconds
  // auto start_time = std::chrono::high_resolution_clock::now();
  // auto start_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
  //                          start_time.time_since_epoch())
  //                          .count();

  // std::cout << "GPU: " << this->gpu_id
  //           << " - First chunk started at: " << start_time_ns << std::endl;
  // }

  // TODO: Send the deduplicated chunk to CPU
  this->timekeeper->snapshot("t1-start", false);
  CHECK_CUDA(cudaMemcpy(
      this->destination_buffer + start_index, this->gpu_data + start_index,
      this->chunk_size * sizeof(int), cudaMemcpyDeviceToHost));
  this->timekeeper->snapshot("t1-end", true);

  // TODO: Check the values and stage them for sending
  this->timekeeper->snapshot("deduplication-start", false);
  // auto start_time = std::chrono::high_resolution_clock::now();
  if (this->deduplicate) {
    for (int i = start_index; i < start_index + this->chunk_size; i++) {
      int checked_value =
          this->cpu_merger->check_value(this->destination_buffer[i]);

      // Tuple is new so we should stage it into the send buffer
      if (checked_value != -1) {
        this->cpu_merger->stage(checked_value);
      }
    }
  } else {
    this->cpu_merger->stage_buffer(
        &this->destination_buffer[start_index], this->chunk_size);
  }

  this->timekeeper->snapshot("deduplication-end", true);
  // auto end_time = std::chrono::high_resolution_clock::now();

  // auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
  //                         end_time - start_time)
  //                         .count();
  // this->timekeeper->add_time("deduplication", elapsed_time);
  // std::cout << "GPU: " << this->gpu_id
  // << " - Chunk finished at: " << elapsed_time << std::endl;
  // std::cout << "GPU: " << this->gpu_id
  //           << " - Number of inserts: " << number_of_inserts
  //           << " for chunk starting at index: " << start_index
  //           << "chunk size: " << this->chunk_size << std::endl;
}

void DistinctMergeGPU::start() {
  // std::cout << std::unitbuf;
  // std::cout << "Starting GPU merger" << std::endl;

  int threads_count = this->tuples_count / this->chunk_size;

  std::thread threads[threads_count];

  for (int i = 0; i < threads_count; i++) {
    threads[i] =
        std::thread(&DistinctMergeGPU::exec, this, i * this->chunk_size);
  }

  for (int i = 0; i < threads_count; i++) {
    threads[i].join();
  }

  // Print timestamp in nanoseconds
  // auto time = std::chrono::high_resolution_clock::now();
  // auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
  //                    time.time_since_epoch())
  //                    .count();
  //
  // std::cout << "GPU: " << this->gpu_id
  //           << " - Last chunk finished at (includes deduplication): " <<
  //           time_ns
  //           << std::endl;
}

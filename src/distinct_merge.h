#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <vector>

#include "RDMA/ucx_rdma_client.h"
#include "RDMA/cuncurrent_hashmap.h"

#define DISTINCT_MERGE_BUFFER_SIZE                                             \
  1024 * 1024 * 256 // WARN: we should use smalle send buffer size
#define DISTINCT_MERGE_BUFFER_THRESHOLD 1024 * 1024
#define DISTINCT_MERGE_SEND_CHUNK_SIZE 1024 * 512

struct DistinctMerge {
private:
  std::map<int, bool> seen_values{};
  ConcurrentHashMap<int, bool> seen_values_concurrent{128};

  int *send_buffer;
  UcxRdmaClient *rdma_client{nullptr};
  int send_buffer_start_index{0};
  int send_buffer_end_index{0};


  std::mutex send_buffer_mutex{};
  std::mutex seen_values_mutex{};

  std::vector<int *> receive_buffers{};
  std::vector<int> receive_buffer_sizes{};

  bool finished{false};

public:
  std::thread sender_thread{};
  bool done_flushing{false};
  DistinctMerge() = default;
  DistinctMerge(const std::vector<int *> &receive_buffers,
                const std::vector<int> &receive_buffer_sizes);

  int check_value(int value);

  bool stage(int value);

  void sender();

  void set_rdma_client(UcxRdmaClient *rdma_client);

  void finish();
};

struct DistinctMergeGPU {
public:
  int gpu_id{0};
  int tuples_count{0};
  int chunk_size{0};
  int *gpu_data{nullptr};
  int *destination_buffer{nullptr};

  DistinctMergeGPU(int gpu_id, int tuples_count, int chunk_size);

  DistinctMerge *cpu_merger;

  void exec(int start_index);

  void start();
};

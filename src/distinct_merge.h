#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <vector>

#include "RDMA/cuncurrent_hashmap.h"
#include "RDMA/ucx_rdma_client.h"
#include "RDMA/timekeeper.h"

struct DistinctMerge {
private:
  std::map<int, bool> seen_values{};
  ConcurrentHashMap<int, bool> seen_values_concurrent{128};
  int send_threshold{0};
  TimeKeeper *timekeeper{nullptr};

  unsigned long send_buffer_threshold{1024 * 1024};

  int *send_buffer;
  UcxRdmaClient *rdma_client{nullptr};
  int send_buffer_start_index{0};
  int send_buffer_end_index{0};

  std::mutex send_buffer_mutex{};
  std::mutex seen_values_mutex{};

  std::vector<int *> receive_buffers{};
  std::vector<unsigned long> receive_buffer_sizes{};

  bool finished{false};

public:
  std::thread sender_thread{};
  bool done_flushing{false};
  DistinctMerge() = default;
  DistinctMerge(const std::vector<int *> &receive_buffers,
                const std::vector<unsigned long> &receive_buffer_sizes,
                unsigned long send_buffer_size,
                unsigned long send_buffer_threshold, TimeKeeper *timekeeper);

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

  float randomness{1.0f};

  TimeKeeper *timekeeper{nullptr};

  bool deduplicate{false};

  bool first_chunk_started{false};

  DistinctMergeGPU(int gpu_id, int tuples_count, int chunk_size, bool deduplicate, TimeKeeper *timekeeper, float randomness);

  DistinctMerge *cpu_merger;

  void exec(int start_index);

  void start();
};

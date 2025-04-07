#include "distinct_merge.h"

#include <thread>

#define DEDUPLICATION_TUPLES_COUNT 1024 * 1024 * 256
#define DEDUPLICATION_CHUNK_SIZE 1024 * 1024

void start_deduplication(DistinctMergeGPU &merger_gpu) {
  merger_gpu.start();
}

int main(int argc, char *argv[]) {

  DistinctMergeGPU merger_gpu1(0, DEDUPLICATION_TUPLES_COUNT,
                               DEDUPLICATION_CHUNK_SIZE);
  DistinctMergeGPU merger_gpu2(1, DEDUPLICATION_TUPLES_COUNT,
                               DEDUPLICATION_CHUNK_SIZE);

  DistinctMerge merger(
      {merger_gpu1.destination_buffer, merger_gpu2.destination_buffer},
      {DEDUPLICATION_TUPLES_COUNT, DEDUPLICATION_TUPLES_COUNT});

  merger_gpu1.cpu_merger = &merger;
  merger_gpu2.cpu_merger = &merger;

  std::thread t1(start_deduplication, std::ref(merger_gpu1));
  std::thread t2(start_deduplication, std::ref(merger_gpu2));

  return 0;
}

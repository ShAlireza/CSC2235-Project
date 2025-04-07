// #include "distinct_merge.h"

#include <iostream>
#include <thread>

#define DEDUPLICATION_TUPLES_COUNT 1024 * 1024 * 256
#define DEDUPLICATION_CHUNK_SIZE 1024 * 1024

// void start_deduplication(DistinctMergeGPU &merger_gpu) {
//   merger_gpu.start();
// }

int main(int argc, char *argv[]) {

  std::cout << "Starting deduplication" << std::endl << std::flush;

  std::cout << "Creating GPU merger 1" << std::endl;
  // DistinctMergeGPU merger_gpu1(0, DEDUPLICATION_TUPLES_COUNT,
  //                              DEDUPLICATION_CHUNK_SIZE);

  std::cout << "Creating GPU merger 2" << std::endl;
  // DistinctMergeGPU merger_gpu2(1, DEDUPLICATION_TUPLES_COUNT,
  //                              DEDUPLICATION_CHUNK_SIZE);

  std::cout << "Creating CPU merger" << std::endl;
  // DistinctMerge merger(
  //     {merger_gpu1.destination_buffer, merger_gpu2.destination_buffer},
  //     {DEDUPLICATION_TUPLES_COUNT, DEDUPLICATION_TUPLES_COUNT});

  // merger_gpu1.cpu_merger = &merger;
  // merger_gpu2.cpu_merger = &merger;

  std::cout << "Starting GPU 1 merger" << std::endl;
  // std::thread t1(start_deduplication, std::ref(merger_gpu1));
  std::cout << "Starting GPU 2 merger" << std::endl;
  // std::thread t2(start_deduplication, std::ref(merger_gpu2));

  return 0;
}

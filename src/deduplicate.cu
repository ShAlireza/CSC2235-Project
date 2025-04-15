#include "RDMA/ucx_rdma_client.h"
#include "distinct_merge.h"

#include <iostream>
#include <thread>

#define DEDUPLICATION_TUPLES_COUNT 1024 * 1024 * 2
#define DEDUPLICATION_CHUNK_SIZE 1024 * 1024
#define DESTINATION_HOST_IP "localhost" // For now

void start_deduplication(DistinctMergeGPU &merger_gpu) { merger_gpu.start(); }

int main(int argc, char *argv[]) {

  int gpu1 = std::stoi(argv[1]);
  int gpu2 = std::stoi(argv[2]);

  std::cout << std::unitbuf;

  std::cout << "Starting deduplication" << std::endl;

  std::cout << "Creating GPU merger 1" << std::endl;
  DistinctMergeGPU merger_gpu1(gpu1, DEDUPLICATION_TUPLES_COUNT,
                               DEDUPLICATION_CHUNK_SIZE);

  std::cout << "Creating GPU merger 2" << std::endl;
  DistinctMergeGPU merger_gpu2(gpu2, DEDUPLICATION_TUPLES_COUNT,
                               DEDUPLICATION_CHUNK_SIZE);

  std::cout << "Creating CPU merger" << std::endl;

  std::cout << (merger_gpu1.destination_buffer == nullptr) << std::endl;
  std::cout << (merger_gpu2.destination_buffer == nullptr) << std::endl;

  std::vector<int *> recv_buffers = {merger_gpu1.destination_buffer,
                                     merger_gpu2.destination_buffer};
  std::vector<int> recv_buffer_sizes = {DEDUPLICATION_TUPLES_COUNT,
                                        DEDUPLICATION_TUPLES_COUNT};

  DistinctMerge merger(recv_buffers, recv_buffer_sizes);

  UcxRdmaClient *rdma_client = new UcxRdmaClient(
      DESTINATION_HOST_IP, DEDUPLICATION_TUPLES_COUNT * 2 * sizeof(int),
      DISTINCT_MERGE_SEND_CHUNK_SIZE * sizeof(int));
  merger.set_rdma_client(rdma_client);
  merger_gpu1.cpu_merger = &merger;
  merger_gpu2.cpu_merger = &merger;

  std::cout << "Starting GPU 1 merger" << std::endl;
  std::thread t1(start_deduplication, std::ref(merger_gpu1));
  std::cout << "Starting GPU 2 merger" << std::endl;
  std::thread t2(start_deduplication, std::ref(merger_gpu2));

  // while(true);

  t1.join();
  t2.join();

  merger.finish();

  std::cout << "Joining the sender thread and closing it..." << std::endl;
  // merger.sender_thread.join();
  while (!merger.done_flushing);

  return 0;
}

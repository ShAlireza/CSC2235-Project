#include "RDMA/ucx_rdma_client.h"
#include "distinct_merge.h"

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <thread>

#define DEDUPLICATION_TUPLES_COUNT 1024 * 1024 * 2
#define DEDUPLICATION_CHUNK_SIZE 1024 * 1024

struct cmd_args {
  unsigned long tuples_count{DEDUPLICATION_TUPLES_COUNT};
  unsigned long chunk_size{DEDUPLICATION_CHUNK_SIZE};
  std::string server_ip{"localhost"};
  int server_port{13337};
  int gpu1{0};
  int gpu2{1};
  unsigned long send_buffer_threshold{0};
};

// Print help message
void print_help() {
  std::cout << "Usage: deduplicate -t <tuples_count> -c <chunk_size> "
               "-s <server_ip> -p <server_port> -1 <gpu1> -2 <gpu2>"
            << std::endl;
  std::cout << "Default values:" << std::endl;
  std::cout << "-t: " << DEDUPLICATION_TUPLES_COUNT << std::endl;
  std::cout << "-c: " << DEDUPLICATION_CHUNK_SIZE << std::endl;
  std::cout << "-s: localhost" << std::endl;
  std::cout << "-p: 13337" << std::endl;
  std::cout << "-1: 0" << std::endl;
  std::cout << "-2: 1" << std::endl;
  std::cout << "-b: " << "chunk_size" << std::endl;
}

// Parse command line arguments
cmd_args parse_args(int argc, char *argv[]) {
  // Parse arguments using getopt or any other library
  cmd_args args{};

  char c{0};
  while ((c = getopt(argc, argv, "t:c:s:p:1:2:b:")) != -1) {
    switch (c) {
    case 't':
      args.tuples_count = std::stoul(optarg);
      break;
    case 'c':
      args.chunk_size = std::stoul(optarg);
      break;
    case 's':
      args.server_ip = optarg;
      break;
    case 'p':
      args.server_port = std::stoi(optarg);
      break;
    case '1':
      args.gpu1 = std::stoi(optarg);
      break;
    case '2':
      args.gpu2 = std::stoi(optarg);
      break;
    case 'b':
      args.send_buffer_threshold = std::stoul(optarg);
      break;
    default:
      print_help();
      exit(EXIT_FAILURE);
    }
  }

  if (args.send_buffer_threshold == 0) {
    args.send_buffer_threshold = args.chunk_size;
  }

  if (args.tuples_count == 0) {
    std::cerr << "Invalid tuples count" << std::endl;
    print_help();
    exit(EXIT_FAILURE);
  }

  return args;
}

void start_deduplication(DistinctMergeGPU &merger_gpu) { merger_gpu.start(); }

int main(int argc, char *argv[]) {
  cmd_args args = parse_args(argc, argv);
  // int gpu1 = std::stoi(argv[1]);
  // int gpu2 = std::stoi(argv[2]);
  //
  // std::string destination_ip = argv[3];

  std::cout << std::unitbuf;

  std::cout << "Starting deduplication" << std::endl;

  std::cout << "Creating GPU merger 1" << std::endl;
  DistinctMergeGPU merger_gpu1(args.gpu1, DEDUPLICATION_TUPLES_COUNT,
                               DEDUPLICATION_CHUNK_SIZE);

  std::cout << "Creating GPU merger 2" << std::endl;
  DistinctMergeGPU merger_gpu2(args.gpu2, DEDUPLICATION_TUPLES_COUNT,
                               DEDUPLICATION_CHUNK_SIZE);

  std::cout << "Creating CPU merger" << std::endl;

  std::cout << (merger_gpu1.destination_buffer == nullptr) << std::endl;
  std::cout << (merger_gpu2.destination_buffer == nullptr) << std::endl;

  std::vector<int *> recv_buffers = {merger_gpu1.destination_buffer,
                                     merger_gpu2.destination_buffer};
  std::vector<int> recv_buffer_sizes = {DEDUPLICATION_TUPLES_COUNT,
                                        DEDUPLICATION_TUPLES_COUNT};

  DistinctMerge merger(recv_buffers, recv_buffer_sizes, args.tuples_count * 2, args.send_buffer_threshold);

  UcxRdmaClient *rdma_client = new UcxRdmaClient(
      args.server_ip, args.server_port, args.tuples_count * 2 * sizeof(int),
      args.chunk_size * sizeof(int));
  merger.set_rdma_client(rdma_client);
  merger_gpu1.cpu_merger = &merger;
  merger_gpu2.cpu_merger = &merger;

  std::cout << "Starting GPU 1 merger" << std::endl;
  std::thread t1(start_deduplication, std::ref(merger_gpu1));
  std::cout << "Starting GPU 2 merger" << std::endl;
  std::thread t2(start_deduplication, std::ref(merger_gpu2));

  t1.join();
  t2.join();

  merger.finish();

  std::cout << "Joining the sender thread and closing it..." << std::endl;
  // merger.sender_thread.join();
  while (!merger.done_flushing)
    ;

  return 0;
}

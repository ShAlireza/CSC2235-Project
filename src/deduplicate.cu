#include "RDMA/timekeeper.h"
#include "RDMA/ucx_rdma_client.h"
#include "distinct_merge.h"

#include <complex>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <netdb.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

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
  std::string peer_ip{""};
  int peer_port{9090};
  bool deduplicate{false};
  float randomness{1.0f};
};

// Print help message
void print_help() {
  std::cout
      << "Usage: deduplicate -t <tuples_count> -c <chunk_size> "
         "-s <server_ip> -p <server_port> -1 <gpu1> -2 <gpu2> -b <buffer_size> "
         "-S <peer_ip> -P <peer_port> -d <enables_deduplication> -r <randomess>"

      << std::endl;
  std::cout << "Default values:" << std::endl;
  std::cout << "-t: " << DEDUPLICATION_TUPLES_COUNT << " Tuples" << std::endl;
  std::cout << "-c: " << DEDUPLICATION_CHUNK_SIZE << " Tuples" << std::endl;
  std::cout << "-s: localhost" << std::endl;
  std::cout << "-p: 13337" << std::endl;
  std::cout << "-1: GPU 0" << std::endl;
  std::cout << "-2: GPU 1" << std::endl;
  std::cout << "-b: " << "chunk_size (# Tuples)" << std::endl;
  std::cout << "-S: <peer_ip>" << std::endl;
  std::cout << "-P: 9090" << std::endl;
  std::cout << "-d: enables deduplication" << std::endl;
  std::cout << "-r: 1.0" << std::endl;
}

// Parse command line arguments
cmd_args parse_args(int argc, char *argv[]) {
  // Parse arguments using getopt or any other library
  cmd_args args{};

  char c{0};
  while ((c = getopt(argc, argv, "t:c:s:p:1:2:b:S:P:dr:")) != -1) {
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
    case 'S':
      args.peer_ip = optarg;
      break;
    case 'P':
      args.peer_port = std::stoi(optarg);
      break;
    case 'd':
      args.deduplicate = true;
      break;
    case 'r':
      args.randomness = std::stof(optarg);
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

int connect_common(const std::string &peer_ip, int peer_port) {
  struct addrinfo hints{}, *res = nullptr;
  int sockfd = -1;
  int optval = 1;

  // prepare the port string
  std::string port_str = std::to_string(peer_port);

  // both client & server use same hints except AI_PASSIVE for server
  hints.ai_family = AF_UNSPEC; // allow IPv4 or IPv6
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = peer_ip.empty()  // if no peer_ip, act as server
                       ? AI_PASSIVE // bind to all local interfaces
                       : 0;

  // getaddrinfo may return multiple candidate addrinfo structs
  int r = getaddrinfo(peer_ip.empty() ? nullptr : peer_ip.c_str(),
                      port_str.c_str(), &hints, &res);
  if (r != 0) {
    std::cerr << "getaddrinfo: " << gai_strerror(r) << "\n";
    return -1;
  }

  for (auto p = res; p; p = p->ai_next) {
    sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (sockfd < 0) {
      // try next
      continue;
    }

    if (!peer_ip.empty()) {
      // ----- CLIENT PATH -----
      if (connect(sockfd, p->ai_addr, p->ai_addrlen) == 0) {
        std::cout << "Connected to " << peer_ip << ":" << peer_port << "\n";
        break; // success!
      }
      std::cerr << "connect: " << strerror(errno) << "\n";
    } else {
      // ----- SERVER PATH -----
      if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval,
                     sizeof(optval)) < 0) {
        std::cerr << "setsockopt: " << strerror(errno) << "\n";
        close(sockfd);
        sockfd = -1;
        continue;
      }
      if (bind(sockfd, p->ai_addr, p->ai_addrlen) < 0) {
        std::cerr << "bind: " << strerror(errno) << "\n";
        close(sockfd);
        sockfd = -1;
        continue;
      }
      if (listen(sockfd, /*backlog=*/1) < 0) {
        std::cerr << "listen: " << strerror(errno) << "\n";
        close(sockfd);
        sockfd = -1;
        continue;
      }

      std::cout << "Server listening on port " << peer_port << "\n";
      int client_fd = accept(sockfd, nullptr, nullptr);
      if (client_fd < 0) {
        std::cerr << "accept: " << strerror(errno) << "\n";
        close(sockfd);
        sockfd = -1;
        continue;
      }
      close(sockfd); // close the listening socket
      sockfd = client_fd;
      std::cout << "Accepted connection\n";
      break; // got one client, done
    }

    // if we get here, something failed — clean up and try next addrinfo
    close(sockfd);
    sockfd = -1;
  }

  freeaddrinfo(res);

  if (sockfd < 0) {
    std::cerr << "Failed to establish "
              << (peer_ip.empty() ? "server" : "client") << " socket\n";
  }
  return sockfd;
}

int barrier(int socketfd) {
  struct pollfd pfd;

  int dummy = 0;
  ssize_t result = 0;

  result = send(socketfd, &dummy, sizeof(dummy), 0);

  if (result < 0) {
    std::cerr << "Error sending data to socket" << std::endl;
    return result;
  }

  pfd.fd = socketfd;
  pfd.events = POLLIN;
  pfd.revents = 0;

  do {
    result = poll(&pfd, 1, 1);
  } while (result == -1);

  result = recv(socketfd, &dummy, sizeof(dummy), MSG_WAITALL);

  return !(result == sizeof(dummy));
}

void start_deduplication(DistinctMergeGPU &merger_gpu) { merger_gpu.start(); }

int main(int argc, char *argv[]) {
  cmd_args args = parse_args(argc, argv);
  // int gpu1 = std::stoi(argv[1]);
  // int gpu2 = std::stoi(argv[2]);
  //
  // std::string destination_ip = argv[3];

  TimeKeeper *timekeeper = new TimeKeeper();

  std::cout << std::unitbuf;

  std::cout << "Starting deduplication" << std::endl;

  std::cout << "Creating GPU merger 1" << std::endl;
  DistinctMergeGPU merger_gpu1(args.gpu1, args.tuples_count, args.chunk_size,
                               args.deduplicate, timekeeper, args.randomness);

  std::cout << "Creating GPU merger 2" << std::endl;
  DistinctMergeGPU merger_gpu2(args.gpu2, args.tuples_count, args.chunk_size,
                              args.deduplicate, timekeeper, args.randomness);

  std::cout << "Creating CPU merger" << std::endl;

  std::cout << (merger_gpu1.destination_buffer == nullptr) << std::endl;
  std::cout << (merger_gpu2.destination_buffer == nullptr) << std::endl;

  std::vector<int *> recv_buffers = {merger_gpu1.destination_buffer,
                                     merger_gpu2.destination_buffer};
  std::vector<unsigned long> recv_buffer_sizes = {args.tuples_count,
                                                  args.tuples_count};

  DistinctMerge merger(recv_buffers, recv_buffer_sizes, args.tuples_count * 2,
                       args.send_buffer_threshold, timekeeper);

  UcxRdmaClient *rdma_client = new UcxRdmaClient(
      args.server_ip, args.server_port, args.tuples_count * 2 * sizeof(int),
      args.chunk_size * sizeof(int), timekeeper);
  merger.set_rdma_client(rdma_client);
  merger_gpu1.cpu_merger = &merger;
  merger_gpu2.cpu_merger = &merger;

  int socketfd = connect_common(args.peer_ip, args.peer_port);

  barrier(socketfd);

  // Print timestamp in nanoseconds
  auto start = std::chrono::high_resolution_clock::now();
  auto nano_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          start.time_since_epoch())
                          .count();

  std::cout << "Starting at " << nano_seconds << " ns" << std::endl;

  std::cout << "Starting GPU 1 merger" << std::endl;
  std::thread thread1(start_deduplication, std::ref(merger_gpu1));
  std::cout << "Starting GPU 2 merger" << std::endl;
  std::thread thread2(start_deduplication, std::ref(merger_gpu2));

  thread1.join();
  thread2.join();

  merger.finish();

  std::cout << "Joining the sender thread and closing it..." << std::endl;
  // merger.sender_thread.join();
  while (!merger.done_flushing)
    ;

  // std::cout << "Closing merger" << std::endl;
  timekeeper->print_history();

  unsigned long t1{0};
  unsigned long t2{0};

  t2 = timekeeper->get_duration("deduplication-end", "deduplication-start");
  t1 = timekeeper->get_duration("t1-end", "t1-start");

  std::cout << "t1: " << t1 << " ns" << std::endl;
  std::cout << "t2: " << t2 << " ns" << std::endl;

  std::cout << "Tuples sent: " << rdma_client->current_offset / sizeof(int)
            << std::endl;

  return 0;
}

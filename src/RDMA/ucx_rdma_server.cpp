#include <arpa/inet.h>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <thread>
#include <ucp/api/ucp.h>
#include <unistd.h>
#include <getopt.h>

struct cmd_args_t {
  unsigned long distinct_merge_buffer_size{1024 * 1024 * 256};
  unsigned long distinct_merge_buffer_threshold{1024 * 1024 * 2};
  unsigned long port{13337};
};

void parse_arguments(int argc, char **argv, cmd_args_t &args) {
  int option;
  while ((option = getopt(argc, argv, "p:b:t:")) != -1) {
    switch (option) {
    case 'p':
      args.port = strtoul(optarg, nullptr, 10);
      break;
    case 'b':
      args.distinct_merge_buffer_size = strtoul(optarg, nullptr, 10);
      break;
    case 't':
      args.distinct_merge_buffer_threshold = strtoul(optarg, nullptr, 10);
      break;
    default:
      fprintf(stderr,
              "Usage: %s [-p port] [-b buffer_size] [-t "
              "threshold]\n",
              argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

cmd_args_t global_args;


#define AM_ID 1
//#define PORT 13337
#define MAX_CLIENTS 2

//#define DISTINCT_MERGE_BUFFER_SIZE                                             \
  1024 * 1024 * 256 // WARN: we should use smalle send buffer size
//#define DISTINCT_MERGE_BUFFER_THRESHOLD 1024 * 1024 * 2

static ucp_ep_h client_eps[MAX_CLIENTS] = {NULL, NULL};
static int client_count = 0;
//static unsigned long PORT = 13337;
//static unsigned long MAX_CLIENTS = 2;
//static unsigned long DISTINCT_MERGE_BUFFER_SIZE =
    //1024 * 1024 * 256; // WARN: we should use smaller send buffer size
//static unsigned long DISTINCT_MERGE_BUFFER_THRESHOLD =
    //1024 * 1024 * 2; // WARN: we should use smaller send buffer size


typedef struct {
  uint64_t remote_addr;
  // rkey will follow this header
} __attribute__((packed)) rdma_info_t;

struct DistinctMergeDest;

struct ucx_server_t {
  ucp_context_h context;
  ucp_worker_h worker;
  ucp_listener_h listener;
  void *rdma_buffer;
  ucp_mem_h memh;
  size_t buffer_size;
  size_t chunk_size;
  int clients_ready;
  std::map<int, bool> seen_values{};
  DistinctMergeDest *merger{nullptr};
  bool initialize_threads{false};
};

struct DistinctMergeDest {
private:
  int *send_buffer;

  int send_buffer_start_index{0};
  int send_buffer_end_index{0};

  std::mutex send_buffer_mutex{};
  std::mutex seen_values_mutex{};

  bool first_chunk_started{false};

  bool finished{false};

public:
  std::map<int, bool> seen_values{};
  int current_offset{0};
  int *destination_buffer;

  bool done_flushing{false};

  DistinctMergeDest() = default;
  DistinctMergeDest(ssize_t send_buffer_size) {
    cudaMallocHost((void **)&this->send_buffer, send_buffer_size);
    cudaMalloc((void **)&this->destination_buffer, send_buffer_size);

    std::thread sender_thread(&DistinctMergeDest::sender,
                              this); // Start the sender thread

    sender_thread.detach();
  };

  int check_value(int value) {
    // WARN: We should remove locking later since its a performance bottleneck
    // (we should use somthing like Intel TBB)

    std::unique_lock<std::mutex> lock(this->seen_values_mutex);

    auto it = seen_values.find(value);
    if (it != seen_values.end()) {
      // INFO: We assume that input data are positive integers
      lock.unlock();
      return -2;
    } else {
      seen_values.emplace(value, true);
      lock.unlock();
      return value;
    }
  }

  bool stage(int value) {
    std::unique_lock<std::mutex> lock(this->send_buffer_mutex);

    // std::cout << value + 4000 << std::endl;
    this->send_buffer[this->send_buffer_end_index++] = value;
    // std::cout << "Sender thread: end index: " << this->send_buffer_end_index
    //           << std::endl;
    // std::cout << "Sender thread: start index: " <<
    // this->send_buffer_start_index
    //           << std::endl;

    lock.unlock();

    return true;
  }

  void sender() {
    // Send data to the dest GPU
    // TODO: this function check the send buffer and sends data whenever it
    // reached the threshold

    std::cout << "In sender thread" << std::endl;
    while (true) {

      // std::cout << "Sender thread: checking send buffer" << std::endl;
      // std::cout << "Sender thread: start index: "
      //           << this->send_buffer_start_index << std::endl;
      // std::cout << "Sender thread: end index: " <<
      // this->send_buffer_end_index
      //           << std::endl;
      int difference =
          std::abs(this->send_buffer_start_index - this->send_buffer_end_index);

      if (difference >= global_args.distinct_merge_buffer_threshold) {
        if (this->first_chunk_started == false) {
          this->first_chunk_started = true;
          // Print timestamp in nanoseconds
          auto time_now = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              time_now.time_since_epoch())
                              .count();
          printf("Sender thread: First chunk started at timestamp %ld\n",
                 duration);
        }
        // std::cout << "Sender thread: Threshold reached - difference: "
        //           << difference << std::endl;
        int *chunk_ptr = &this->send_buffer[this->send_buffer_start_index];
        int chunk_bytes = difference * sizeof(int);

        cudaMemcpy(this->destination_buffer + current_offset, chunk_ptr,
                   difference * sizeof(int), cudaMemcpyHostToDevice);

        current_offset += difference;

        this->send_buffer_start_index += difference;
        difference = std::abs(this->send_buffer_start_index -
                              this->send_buffer_end_index);
      }

      if (this->finished) {

        std::cout << "Sender thread: checking send buffer" << std::endl;
        std::cout << "Sender thread: start index: "
                  << this->send_buffer_start_index << std::endl;
        std::cout << "Sender thread: end index: " << this->send_buffer_end_index
                  << std::endl;
        std::cout << "Sender thread: Sender flushing - difference: "
                  << difference << std::endl;
        std::cout << "Sender thread finished" << std::endl;

        if (difference > 0) {
          int *chunk_ptr = &this->send_buffer[this->send_buffer_start_index];
          int chunk_bytes = difference * sizeof(int);

          cudaMemcpy(this->destination_buffer + current_offset, chunk_ptr,
                     difference * sizeof(int), cudaMemcpyHostToDevice);

          current_offset += difference;
          this->send_buffer_start_index += difference;
        }

        // Print timestamp in nanoseconds
        auto time_now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            time_now.time_since_epoch())
                            .count();
        printf("Sender thread: Flushing data to GPU at timestamp %ld\n", duration);
        this->done_flushing = true;
        break;
      }
    }
  }

  void finish() { this->finished = true; }
};

void receiver_thread(int *buffer, DistinctMergeDest *merger, bool verbose,
                     int client_id) {
  int old_counter = 0;
  // printf("Buffer addr: %lu\n", (unsigned long)buffer);

  while (1) {
    int counter = buffer[0];

    if (counter != old_counter) {
      if (counter == -1) {
        // Print timestamp in nanoseconds
        auto time_now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            time_now.time_since_epoch())
                            .count();
        printf(
            "[%d]RDMA Server: Received end of stream signal at timestamp %ld\n",
            client_id, duration);

        auto start_time = std::chrono::high_resolution_clock::now();
        while (buffer[1 + old_counter] != 0 && buffer[1 + old_counter] != -1) {
          int check_value = merger->check_value(buffer[1 + old_counter]);
          if (verbose) {
            printf("%d", 1000 + buffer[1 + old_counter]);
            printf("x%d ", check_value);
          }
          if (check_value != -2) {
            merger->stage(check_value);
          }
          old_counter++;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       end_time - start_time)
                       .count();
        printf("[%d]RDMA Server: deduplication took %ld\n", client_id,
               duration);
        // printf("\n");

        // int tuples_count = 0;
        break;
      } else {
        // printf("Server: Received new data from client %d\n", buffer[0]);
        // Process the data
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = old_counter; i < counter; i++) {
          int check_value = merger->check_value(buffer[1 + i]);
          if (check_value != -2) {
            merger->stage(check_value);
          }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            end_time - start_time)
                            .count();
        printf("[%d]RDMA Server: deduplication took %ld\n", client_id,
               duration);
        old_counter = counter;
      }
    }
  }
}
static int init_worker(ucp_context_h ucp_context, ucp_worker_h *ucp_worker) {
  ucp_worker_params_t worker_params;
  ucs_status_t status;
  int ret = 0;

  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

  status = ucp_worker_create(ucp_context, &worker_params, ucp_worker);
  if (status != UCS_OK) {
    fprintf(stderr, "failed to ucp_worker_create (%s)\n",
            ucs_status_string(status));
    ret = -1;
  }
  return ret;
}

void send_cb(void *request, ucs_status_t status, void *user_data) {
  printf("Client: AM message sent successfully (status = %s)\n",
         ucs_status_string(status));
}

void handle_request(void *req, const char *label, ucp_worker_h worker) {
  if (UCS_PTR_IS_PTR(req)) {
    while (ucp_request_check_status(req) == UCS_INPROGRESS) {
      printf("Progressing %s\n", label);
      ucp_worker_progress(worker);
    }
    printf("Freeing %s request %p\n", label, req);
    ucp_request_free(req);
  } else if (!UCS_STATUS_IS_ERR((ucs_status_t)(uintptr_t)req)) {
    send_cb(NULL, (ucs_status_t)(uintptr_t)req, NULL);
  } else {
    fprintf(stderr, "%s request failed: %s\n", label,
            ucs_status_string((ucs_status_t)(uintptr_t)req));
  }
}

void send_rkey_to_client(ucp_ep_h ep, ucx_server_t *server,
                         size_t offset) { // ADDED OFFSET
  void *rkey_buffer = NULL;
  size_t rkey_size = 0;

  printf("Pack the rkey for client\n");
  ucs_status_t status =
      ucp_rkey_pack(server->context, server->memh, &rkey_buffer, &rkey_size);
  if (status != UCS_OK) {
    fprintf(stderr, "ucp_rkey_pack failed: %s\n", ucs_status_string(status));
    return;
  }

  uint64_t addr_offset = (uint64_t)(server->rdma_buffer) + offset; // NEW
  size_t msg_size = sizeof(rdma_info_t); // + rkey_size;
  char *msg = (char *)malloc(msg_size);
  rdma_info_t *info = (rdma_info_t *)msg;
  info->remote_addr = addr_offset; // NEW
  // memcpy(msg + sizeof(rdma_info_t), rkey_buffer, rkey_size);
  printf("Message size is %ld, rkey size is %ld\n", msg_size, rkey_size);
  printf("Remote addr is %ld\n", info->remote_addr);

  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK,
                               .user_data = NULL};
  param.cb.send = send_cb;

  // char *txt = "Hello, UCX!";
  char *x = (char *)malloc(sizeof(uint64_t));
  memcpy(x, &info->remote_addr, sizeof(uint64_t));
  printf("Sending rkey and remote_addr to client\n");
  void *req1 = ucp_am_send_nbx(ep, AM_ID, NULL, 0, x, sizeof(uint64_t), &param);
  void *req2 =
      ucp_am_send_nbx(ep, AM_ID, NULL, 0, rkey_buffer, rkey_size, &param);
  handle_request(req1, "remote_addr", server->worker);
  handle_request(req2, "rkey", server->worker);

  ucp_rkey_buffer_release(rkey_buffer);
  free(msg);
  printf("Server: Sent rkey and remote_addr to client\n");
}

ucs_status_t am_recv_cb(void *arg, const void *header, size_t header_length,
                        void *data, size_t length,
                        const ucp_am_recv_param_t *param) {

  ucx_server_t *server = (ucx_server_t *)arg;
  if (!server) {
    fprintf(stderr, "Error: server is NULL!\n");
    return UCS_ERR_INVALID_PARAM;
  }
  printf("Server received AM: %s\n", (char *)data);

  char *token = strtok((char *)data, " ");
  if (token != NULL) {
    server->chunk_size = strtoul(token, NULL, 10);
    token = strtok(NULL, " ");
    if (token != NULL) {
      server->buffer_size = strtoul(token, NULL, 10);
    }
  }

  server->clients_ready++; // NEW

  printf("Server: Received chunk size %ld and buffer size %ld\n",
         server->chunk_size, server->buffer_size);

  // Only allocate the buffer if both clients are ready NEW

  if (server->clients_ready == MAX_CLIENTS) { // NEW
    size_t total_size = 2 * server->buffer_size;
    server->rdma_buffer =
        calloc(1, total_size + 2 * sizeof(int) +
                      sizeof(int)); // 2 * sizeof(int) is because we are
                                    // holding a counter per sender to
                                    // notify server about new data arrival.
    // server->send_buffer = calloc(1, total_size); // NEW
    // Note that we still allocate total size, which might be too much,
    // but we dont know how many duplicates there will be, so its fine

    // std::memset(server->rdma_buffer, 0, total_size + 2 * sizeof(int));
    // std::memset(server->send_buffer, 0, total_size);
    //
    // std::memset((int *)server->rdma_buffer, 0, sizeof(int));
    // std::memset((int *)server->rdma_buffer + server->buffer_size +
    // sizeof(int), 0,
    //        sizeof(int));
    ucp_mem_map_params_t mmap_params = {.field_mask =
                                            UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                            UCP_MEM_MAP_PARAM_FIELD_LENGTH,
                                        .address = server->rdma_buffer,
                                        .length = total_size + 2 * sizeof(int)};

    ucs_status_t status =
        ucp_mem_map(server->context, &mmap_params, &server->memh);

    if (status != UCS_OK) {
      fprintf(stderr, "ucp_mem_map failed: %s\n", ucs_status_string(status));
      return status;
    }
    printf("Server RDMA buffer allocated at %p\n", server->rdma_buffer);
    printf("Server RDMA buffer size is %ld\n", total_size);
    printf("Server: Both clients are ready, buffer allocated\n");

    // Now send the rkey back to the clients
    for (int i = 0; i < MAX_CLIENTS; i++) { // NEW
      if (client_eps[i] != NULL) {
        send_rkey_to_client(
            client_eps[i], server,
            i * (server->buffer_size + sizeof(int))); // first index is used for
                                                      // the counter
      }
    }

    server->initialize_threads = true;
  }

  return UCS_OK;
}

void on_connection(ucp_conn_request_h conn_request, void *arg) {
  ucp_worker_h worker = (ucp_worker_h)arg;
  ucp_ep_params_t ep_params = {.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST,
                               .conn_request = conn_request};
  ucp_ep_create(worker, &ep_params, &client_eps[client_count]);
  client_count++;
  printf("Server: client connected\n");
}

int start_ucx_server(const cmd_args_t &args) {

  ucx_server_t *server = new ucx_server_t();

  if (!server) {
    fprintf(stderr, "Failed to allocate ucx_server_t\n");
    return -1;
  }
  ucs_status_t status;

  // Init context
  ucp_params_t params = {.field_mask = UCP_PARAM_FIELD_FEATURES,
                         .features = UCP_FEATURE_AM | UCP_FEATURE_RMA};
  status = ucp_init(&params, NULL, &(server->context));
  if (status != UCS_OK) {
    fprintf(stderr, "failed to ucp_init (%s)\n", ucs_status_string(status));
    return -1;
  }
  printf("Context initialized\n");

  if (init_worker(server->context, &(server->worker)) != 0) {
    fprintf(stderr, "failed to init_worker\n");
    ucp_cleanup(server->context);
    return -1;
  }
  printf("Worker initialized\n");

  // Set AM callback
  // ASSUMPTION: First message received by the server will be the chunk size and
  // buffer size The server then immediately sends the rkey to the client
  ucp_am_handler_param_t am_param = {.field_mask =
                                         UCP_AM_HANDLER_PARAM_FIELD_ID |
                                         UCP_AM_HANDLER_PARAM_FIELD_CB |
                                         UCP_AM_HANDLER_PARAM_FIELD_ARG,
                                     .id = AM_ID,
                                     .cb = am_recv_cb,
                                     .arg = server};
  ucp_worker_set_am_recv_handler(server->worker, &am_param);

  // Listener
  struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(args.port)};
  addr.sin_addr.s_addr = INADDR_ANY;
  ucp_listener_params_t listener_params = {
      .field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                    UCP_LISTENER_PARAM_FIELD_CONN_HANDLER,
      .sockaddr = {.addr = (struct sockaddr *)&addr, .addrlen = sizeof(addr)},
      .conn_handler = {.cb = on_connection, .arg = server->worker}};
  ucp_listener_create(server->worker, &listener_params, &(server->listener));
  printf("Server is listening on port %d\n", args.port);

  while (1) {
    ucp_worker_progress(server->worker);

    // if (server->rdma_buffer != NULL) {
    //   int *buffer = (int *)server->rdma_buffer;
    //   size_t entries_per_buffer = server->buffer_size / sizeof(int);
    //
    //   printf("=== Buffer 1 ===\n");
    //   for (size_t i = 0; i < entries_per_buffer; i++) {
    //     printf("%d ", buffer[i]);
    //   }
    //   printf("\n");
    //
    //   printf("=== Buffer 2 ===\n");
    //   for (size_t i = 0; i < entries_per_buffer; i++) {
    //     printf("%d ", buffer[entries_per_buffer + i]);
    //   }
    //   printf("\n");
    //
    //   printf("------------------------------------------------------------\n");
    // }
    //
    // Check if the last element in both halves is non-zero
    if (server->initialize_threads) {
      unsigned long client2_addr = (unsigned long)(server->rdma_buffer) +
                                   server->buffer_size + sizeof(int);

      DistinctMergeDest *merger =
          new DistinctMergeDest(2 * server->buffer_size);
      server->merger = merger;

      std::thread client1_receiver(receiver_thread, (int *)server->rdma_buffer,
                                   merger, false, 0);
      std::thread client2_receiver(receiver_thread, (int *)client2_addr, merger,
                                   false, 1);

      client1_receiver.join();
      client2_receiver.join();

      printf("Both clients finished sending data\n");

      server->merger->finish();

      break;
    }
    // if (server->rdma_buffer != NULL &&
    //     (((int *)server->rdma_buffer)[0] == -1) &&
    //     (((int *)server->rdma_buffer)[server->buffer_size / sizeof(int) + 1]
    //     ==
    //      -1)) {
    // sleep(2);
    // printf("Both clients finished sending data\n");
    // server->merger->finish();
    // break;
    // Now that they are full, we should iterate over the buffer and use the
    // seen values map to check if we've seen the value before. If the value
    // is unique, we can put it into the send_buffer. Otherwise, we can ignore
    // it.
    // int *input = (int *)server->rdma_buffer;
    // int *send_buffer = (int *)server->send_buffer;
    // int total_entries = 2 * (server->buffer_size / sizeof(int));
    // printf("Total entries: %d\n", total_entries);
    // int unique_index = 0;

    // for (size_t i = 0; i < total_entries; i++) {
    //   int value = input[i];
    //   printf("Value: %d\n", value);
    //   if (server->seen_values.find(value) ==
    //       server->seen_values.end()) { // if not found
    //     printf("Value not found in map\n");
    //     server->seen_values.emplace(value, true);
    //     printf("Adding value to map\n");
    //     // Instead of writing at send_buffer[i], use unique_index
    //     send_buffer[unique_index] = value;
    //     unique_index++;
    //     printf("Unique value: %d stored at index %d\n", value,
    //            unique_index - 1);
    //   }
    // }
    // }

    usleep(1000);
  }

  while (!server->merger->done_flushing)
    ;

  int *verification;
  cudaMallocHost(&verification, 2 * server->buffer_size);
  cudaMemcpy(verification, server->merger->destination_buffer,
             2 * server->buffer_size, cudaMemcpyDeviceToHost);

  std::cout << "Offset value: " << server->merger->current_offset << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << verification[i] << " ";
  }
  std::cout << std::endl;

  // Print seen values map from merger
  // for (auto it : server->merger->seen_values) {
  //   std::cout << "Key: " << it.first << ", Value: " << it.second <<
  //   std::endl;
  // }

  // print last 10 numbers
  for (int i = server->merger->current_offset - 10;
       i < server->merger->current_offset; i++) {
    std::cout << verification[i] << " ";
  }
  std::cout << std::endl;

  // ucp_mem_unmap(server->context, server->memh);
  // free(server->rdma_buffer);
  ucp_listener_destroy(server->listener);
  ucp_worker_destroy(server->worker);
  ucp_cleanup(server->context);
  // free(server);

  return 0;
}

int main(int argc, char **argv) {
  cmd_args_t args;
  parse_arguments(argc, argv, args);
  return start_ucx_server(args);
}

#include <arpa/inet.h>
#include <queue>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <ucp/api/ucp.h>
#include <unistd.h>

#define AM_ID 1
#define PORT 13337
#define RDMA_MSG "This is RDMA data from client!"
#define BUFFER_SIZE 128
#define CHUNK_SIZE 16

typedef struct {
  uint64_t remote_addr;
  // rkey will follow this header
} __attribute__((packed)) rdma_info_t;

typedef struct {
  ucp_context_h context;
  ucp_worker_h worker;
  ucp_ep_h ep;
  ucp_rkey_h rkey;
  uint64_t remote_addr;
  int rkey_received;
  std::queue<void *> requests;
  bool finished;
  int *rdma_data;
} ucx_client_t;

//static ucp_rkey_h global_rkey = NULL;
//static uint64_t global_remote_addr = 0;
//static int rkey_received = 0;

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
  int id = (int)(uintptr_t)user_data;
  printf("Client: AM message %d sent successfully (status = %s)\n", id,
         ucs_status_string(status));
}

void rdma_cb(void *request, ucs_status_t status, void *user_data) {
  int id = (int)(uintptr_t)user_data;
  printf("Client: RDMA write %d completed (status = %s)\n", id,
         ucs_status_string(status));
}

ucs_status_t rkey_recv_cb(void *arg, const void *header, size_t header_length,
                          void *data, size_t length,
                          const ucp_am_recv_param_t *param) {
  ucx_client_t *client = (ucx_client_t *)arg;

  printf("Header size is %ld\n", header_length);
  if (data == NULL) {
    fprintf(stderr, "Invalid RDMA info message\n");
    return UCS_OK;
  }
  size_t rkey_size;
  void *rkey_buf;
  printf("Message size is %ld\n", length);
  if (length == sizeof(uint64_t)) { // then we know it's the remote_addr
    printf("Length for remote addr(and sizeof(uint64)) is %ld\n", length);
    printf("x is %ld\n", *((uint64_t *)data));
    client->remote_addr = *((uint64_t *)data);
  } else if (length > sizeof(uint64_t)) { // then we know it's the rkey
    printf("Length for rkey is %ld\n", length);
    rkey_size = length;
    rkey_buf = data;

    ucs_status_t status = ucp_ep_rkey_unpack(client->ep, rkey_buf, &client->rkey);
    if (status != UCS_OK) {
      fprintf(stderr, "Failed to unpack rkey (%s)\n",
              ucs_status_string(status));
      return status;
    }

    client->rkey_received = 1;

    return UCS_OK;
  } else {
    fprintf(stderr, "Invalid RDMA info message\n");
    return UCS_ERR_INVALID_PARAM;
  }

  if (client->rkey != NULL) {
    ucp_rkey_destroy(client->rkey);
    client->rkey = NULL;
  }

  fprintf(stderr, "Client: Received remote_addr = 0x%lx and unpacked rkey\n",
          client->remote_addr);
  return UCS_OK;
}

// This function creates a buffer and puts some random data in it
int *generate_random_data(int size) {
  int *buffer = (int *)malloc(size);
  if (buffer == NULL) {
    fprintf(stderr, "Failed to allocate buffer\n");
    return NULL;
  }
  for (int i = 0; i < size / sizeof(int); i++) {
    buffer[i] = i;
  }

  return buffer;
}

void sender(std::queue<void *> *requests, int threshold, ucp_worker_h worker,
            ucp_request_param_t put_param, bool *finished) {

  while (true) {
    if (*finished) {
      break;
    }
    if (requests->size() >= threshold) {
      void *req = requests->front();
      requests->pop();
      printf("Put request is %p\n", req);
      if (UCS_PTR_IS_PTR(req)) {
        while (ucp_request_check_status(req) == UCS_INPROGRESS)
          ucp_worker_progress(worker);
        printf("Freeing PUT request %p\n", req);
        ucp_request_free(req);
      } else if (!UCS_STATUS_IS_ERR((ucs_status_t)(uintptr_t)req)) {
        rdma_cb(NULL, (ucs_status_t)(uintptr_t)req, put_param.user_data);
      } else {
        fprintf(stderr, "Request failed: %s\n",
                ucs_status_string((ucs_status_t)(uintptr_t)req));
      }
    }
  }

  printf("Sender thread finished\n");
  printf("Flushing the requests queue\n");
  printf("Before Loop: Requests size is %ld\n", requests->size());
  while (requests->size() > 0) {
    printf("Requests size is %ld\n", requests->size());
    void *req = requests->front();
    requests->pop();
    printf("Put request is %p\n", req);
    if (UCS_PTR_IS_PTR(req)) {
      while (ucp_request_check_status(req) == UCS_INPROGRESS)
        ucp_worker_progress(worker);
      printf("Freeing PUT request %p\n", req);
      ucp_request_free(req);
    } else if (!UCS_STATUS_IS_ERR((ucs_status_t)(uintptr_t)req)) {
      rdma_cb(NULL, (ucs_status_t)(uintptr_t)req, put_param.user_data);
    } else {
      fprintf(stderr, "Request failed: %s\n",
              ucs_status_string((ucs_status_t)(uintptr_t)req));
    }
  }
  printf("Flushed the requests queue\n");
}

void send_BF_sizes(ucp_ep_h ep, ucp_worker_h worker) {
  // send the chunk size value to the server using AM
  // create a string message with the chunk size and buffer size to send. There should be a seperator between the two to parse it on the server side
  char *msg = (char *)malloc(64);
  snprintf(msg, 64, "%d %d", CHUNK_SIZE, BUFFER_SIZE);
  printf("Sending chunk size %d and buffer size %d\n", CHUNK_SIZE,
         BUFFER_SIZE);
  ucp_request_param_t am_chunk_param = {
      .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA,
      .user_data = (void *)(uintptr_t)0};
  am_chunk_param.cb.send = send_cb;
  void *am_req = ucp_am_send_nbx(ep, AM_ID, NULL, 0, msg, strlen(msg) + 1,
                                 &am_chunk_param);
  printf("AM request is %p\n", am_req);
  // make sure to block until the AM is sent
  if (UCS_PTR_IS_PTR(am_req)) {
    while (ucp_request_check_status(am_req) == UCS_INPROGRESS)
      ucp_worker_progress(worker);
    printf("Freeing AM request %p\n", am_req);
    ucp_request_free(am_req);
  } else if (!UCS_STATUS_IS_ERR((ucs_status_t)(uintptr_t)am_req)) {
    send_cb(NULL, (ucs_status_t)(uintptr_t)am_req, am_chunk_param.user_data);
  } else {
    fprintf(stderr, "Request failed: %s\n",
            ucs_status_string((ucs_status_t)(uintptr_t)am_req));
  }
}

int start_ucx_client(const char *server_ip) {

  bool finished = false;
  int threshold = 100;

  ucx_client_t *client = new ucx_client_t();

  // Init context
  ucp_params_t params = {.field_mask = UCP_PARAM_FIELD_FEATURES,
                         .features = UCP_FEATURE_AM | UCP_FEATURE_RMA};
  ucp_init(&params, NULL, &client->context);

  // Init worker
  if (init_worker(client->context, &client->worker) != 0) {
    fprintf(stderr, "failed to init_worker\n");
    ucp_cleanup(client->context);
    return -1;
  }
  ucp_request_param_t put_param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                                   UCP_OP_ATTR_FIELD_USER_DATA,
                                   .user_data = (void *)(uintptr_t)0};
  put_param.cb.send = rdma_cb;

  // Start a thread to check sending queue
  std::thread sender_thread(sender, &client->requests, threshold, client->worker, put_param,
                            &finished);

  // Create endpoint
  struct sockaddr_in server_addr = {.sin_family = AF_INET,
                                    .sin_port = htons(PORT)};
  inet_pton(AF_INET, server_ip, &server_addr.sin_addr);
  ucp_ep_params_t ep_params = {
      .field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
                    UCP_EP_PARAM_FIELD_ERR_HANDLER,
      .flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER,
      .sockaddr = {.addr = (struct sockaddr *)&server_addr,
                   .addrlen = sizeof(server_addr)}};
  ep_params.err_handler.cb = NULL;
  ep_params.err_handler.arg = NULL;
  ucp_ep_create(client->worker, &ep_params, &client->ep);

  // Register AM handler to receive rkey from server
  ucp_am_handler_param_t am_param = {.field_mask =
                                         UCP_AM_HANDLER_PARAM_FIELD_ID |
                                         UCP_AM_HANDLER_PARAM_FIELD_CB |
                                         UCP_AM_HANDLER_PARAM_FIELD_ARG,
                                     .id = AM_ID,
                                     .cb = rkey_recv_cb,
                                     .arg = client};
  ucp_worker_set_am_recv_handler(client->worker, &am_param);

  // Send the Chunk Size and Buffer Size Metadata to the server
  send_BF_sizes(client->ep, client->worker);

  // Wait until rkey is received
  while (!client->rkey_received) {
    ucp_worker_progress(client->worker);
  }

  int *rdma_data = generate_random_data(BUFFER_SIZE);
  rdma_data[BUFFER_SIZE - 1] = '\0'; // Null-terminate the string

  int chunks_count = BUFFER_SIZE / CHUNK_SIZE;
  for (int i = 0; i < chunks_count; i++) {
    // Print the rdma data
    printf("Client: Sending RDMA data chunk: ");
    for (int j = i * CHUNK_SIZE / sizeof(int);
         j < i * CHUNK_SIZE / sizeof(int) + CHUNK_SIZE / sizeof(int); j++) {
      printf("%d ", rdma_data[j]);
    }
    printf("\n");

    put_param.user_data = (void *)(uintptr_t)i;
    printf("Sending chunk %d\n", i);

    void *put_req =
        ucp_put_nbx(client->ep, rdma_data + i * CHUNK_SIZE / sizeof(int), CHUNK_SIZE,
                    client->remote_addr + i * CHUNK_SIZE,
                    client->rkey, &put_param);
    printf("Put request is %p\n", put_req);
    client->requests.push(put_req);
    printf("Requests size is %ld\n", client->requests.size());
  }

  printf("All data sent\n");
  finished = true;

  sender_thread.join();

  printf("Waiting for sender thread to finish\n");
  free(rdma_data);

  if (client->rkey != NULL) {
    ucp_rkey_destroy(client->rkey);
  }
  ucp_ep_destroy(client->ep);
  ucp_worker_destroy(client->worker);
  ucp_cleanup(client->context);
  delete client;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <server_ip>\n", argv[0]);
    return 1;
  }
  const char *server_ip = argv[1];
  start_ucx_client(server_ip);
  return 0;
}

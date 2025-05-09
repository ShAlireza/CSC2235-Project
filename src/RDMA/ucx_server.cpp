#include <arpa/inet.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <ucp/api/ucp.h>
#include <unistd.h>

#define AM_ID 1
#define PORT 13337
static ucp_ep_h client_ep = NULL;
// #define BUFFER_SIZE 128
// #define CHUNK_SIZE 16

// Chunk size will actually be receied from the client as an initial message. so
// we need to replace CHUNK_SIZE with the value received from the client The
// next line will be a global variable which will be modified by the client

typedef struct {
  uint64_t remote_addr;
  // rkey will follow this header
} __attribute__((packed)) rdma_info_t;

typedef struct {
  ucp_context_h context;
  ucp_worker_h worker;
  ucp_listener_h listener;
  ucp_ep_h client_ep;
  void *rdma_buffer;
  ucp_mem_h memh;
  size_t buffer_size;
  size_t chunk_size;
} ucx_server_t;

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

void send_rkey_to_client(ucp_ep_h ep, ucx_server_t *server) {
  void *rkey_buffer = NULL;
  size_t rkey_size = 0;

  printf("Pack the rkey for client\n");
  ucs_status_t status =
      ucp_rkey_pack(server->context, server->memh, &rkey_buffer, &rkey_size);
  if (status != UCS_OK) {
    fprintf(stderr, "ucp_rkey_pack failed: %s\n", ucs_status_string(status));
    return;
  }
  size_t msg_size = sizeof(rdma_info_t); // + rkey_size;
  char *msg = (char *)malloc(msg_size);
  rdma_info_t *info = (rdma_info_t *)msg;
  info->remote_addr = (uint64_t)(server->rdma_buffer);
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
  printf("Server: Received chunk size %ld and buffer size %ld\n",
         server->chunk_size, server->buffer_size);

  // Allocate and register real RDMA buffer
  server->rdma_buffer = calloc(1, server->buffer_size);
  // print the size and address of the buffer
  printf("Server RDMA buffer allocated at %p\n", server->rdma_buffer);
  printf("Server RDMA buffer size is %ld\n", server->buffer_size);
  ucp_mem_map_params_t mmap_params = {.field_mask =
                                          UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                          UCP_MEM_MAP_PARAM_FIELD_LENGTH,
                                      .address = server->rdma_buffer,
                                      .length = server->buffer_size};
  ucs_status_t status =
      ucp_mem_map(server->context, &mmap_params, &server->memh);
  if (status != UCS_OK) {
    fprintf(stderr, "ucp_mem_map failed: %s\n", ucs_status_string(status));
    return status;
  }

  // Now send the rkey back to the client
  send_rkey_to_client(client_ep, server);

  return UCS_OK;
}

void on_connection(ucp_conn_request_h conn_request, void *arg) {
  ucp_worker_h worker = (ucp_worker_h)arg;
  ucp_ep_params_t ep_params = {.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST,
                               .conn_request = conn_request};
  ucp_ep_create(worker, &ep_params, &client_ep);
  printf("Server: client connected\n");
}

int start_ucx_server(uint16_t port) {

  ucx_server_t *server = (ucx_server_t *)calloc(1, sizeof(ucx_server_t));
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
  struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(PORT)};
  addr.sin_addr.s_addr = INADDR_ANY;
  ucp_listener_params_t listener_params = {
      .field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                    UCP_LISTENER_PARAM_FIELD_CONN_HANDLER,
      .sockaddr = {.addr = (struct sockaddr *)&addr, .addrlen = sizeof(addr)},
      .conn_handler = {.cb = on_connection, .arg = server->worker}};
  ucp_listener_create(server->worker, &listener_params, &(server->listener));
  printf("Server is listening on port %d\n", PORT);
  while (1) {
    ucp_worker_progress(server->worker);
    // if (server->rdma_buffer != NULL) {
      // for (int i = 0; i < server->buffer_size / sizeof(int); i++) {
      //   printf("%d ", ((int *)server->rdma_buffer)[i]);
      // }
      // printf("\n");
      // printf("------------------------------------------------------------\n");
    // }
    if (server->rdma_buffer != NULL &&
        ((int *)server->rdma_buffer)[server->buffer_size / sizeof(int) - 1] !=
            0) {
      for (int i = 0; i < 40 / sizeof(int); i++) {
        printf("%d ", ((int *)server->rdma_buffer)[i]);
      }
      // Print the last 10 integers
      for (int i = server->buffer_size / sizeof(int) - 10;
           i < server->buffer_size / sizeof(int); i++) {
        printf("%d ", ((int *)server->rdma_buffer)[i]);
      }
      printf("\n");
      printf("------------------------------------------------------------\n");

      sleep(2);
      printf("Buffer is full\n");
      break;
    }
    usleep(1000);
  }

  ucp_mem_unmap(server->context, server->memh);
  free(server->rdma_buffer);
  ucp_listener_destroy(server->listener);
  ucp_worker_destroy(server->worker);
  ucp_cleanup(server->context);
  free(server);
  return 0;
}

int main() {
  // Start the server
  start_ucx_server(PORT);
}

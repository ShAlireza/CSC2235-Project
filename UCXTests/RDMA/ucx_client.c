#include <arpa/inet.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucp/api/ucp.h>
#include <unistd.h>

#define AM_ID 1
#define PORT 13337
#define RDMA_MSG "This is RDMA data from client!"
#define RDMA_MSG_SIZE 128

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
  if (request != NULL) {
    ucp_request_free(request);
  }
}

void rdma_cb(void *request, ucs_status_t status, void *user_data) {
  int id = (int)(uintptr_t)user_data;
  printf("Client: RDMA write %d completed (status = %s)\n", id,
         ucs_status_string(status));
  if (request != NULL) {
    ucp_request_free(request);
  }
}

typedef struct {
  uint64_t remote_addr;
  // rkey will follow this header
} __attribute__((packed)) rdma_info_t;

static ucp_rkey_h global_rkey = NULL;
static uint64_t global_remote_addr = 0;
static int rkey_received = 0;

ucs_status_t rkey_recv_cb(void *arg, const void *header, size_t header_length,
                          void *data, size_t length,
                          const ucp_am_recv_param_t *param) {
  ucp_ep_h ep = (ucp_ep_h)arg;

  // if (length < sizeof(rdma_info_t)) {
  //   fprintf(stderr, "Invalid RDMA info message\n");
  //   return UCS_OK;
  // }

  printf("Header size is %ld\n", header_length);
  if (data == NULL) {
    fprintf(stderr, "Invalid RDMA info message\n");
    return UCS_OK;
  }
  size_t rkey_size;
  void *rkey_buf;
  // int *x = (int *)data;
  // const rdma_info_t *info = (const rdma_info_t *)data;
  // global_remote_addr = info->remote_addr;
  printf("Message size is %ld\n", length);
  // printf("Remote addr is %ld\n", info->remote_addr);
  if (length == sizeof(uint64_t)) {
    printf("x is %ld\n", *((uint64_t *)data));
    global_remote_addr = *((uint64_t *)data);
  } else {
    rkey_size = length;
    rkey_buf = data;
    fprintf(stderr, "Invalid RDMA info message\n");

    ucs_status_t status = ucp_ep_rkey_unpack(ep, rkey_buf, &global_rkey);
    if (status != UCS_OK) {
      fprintf(stderr, "Failed to unpack rkey (%s)\n",
              ucs_status_string(status));
      return status;
    }

    rkey_received = 1;

    return UCS_OK;
  }
  // printf("Message: %s\n", (char *)data);

  // char *bytes = (char *)data;
  // for (int i = 0; i < length; i++) {
  //   printf("%d ", bytes[i]);
  // }

  if (global_rkey != NULL) {
    ucp_rkey_destroy(global_rkey);
    global_rkey = NULL;
  }

  fprintf(stderr, "Client: Received remote_addr = 0x%lx and unpacked rkey\n",
          global_remote_addr);
  return UCS_OK;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <server_ip>\n", argv[0]);
    return 1;
  }

  const char *server_ip = argv[1];
  ucp_context_h context;
  ucp_worker_h worker;
  ucp_ep_h ep;

  // Init context
  ucp_params_t params = {.field_mask = UCP_PARAM_FIELD_FEATURES,
                         .features = UCP_FEATURE_AM | UCP_FEATURE_RMA};
  ucp_init(&params, NULL, &context);

  // Init worker
  if (init_worker(context, &worker) != 0) {
    fprintf(stderr, "failed to init_worker\n");
    ucp_cleanup(context);
    return -1;
  }

  // Create endpoint
  struct sockaddr_in server_addr = {.sin_family = AF_INET,
                                    .sin_port = htons(PORT)};
  inet_pton(AF_INET, server_ip, &server_addr.sin_addr);
  ucp_ep_params_t ep_params = {
      .field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
                    UCP_EP_PARAM_FIELD_ERR_HANDLER,
      .flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER,
      .sockaddr = {.addr = (struct sockaddr *)&server_addr,
                   .addrlen = sizeof(server_addr)},
      .err_handler.cb = NULL,
      .err_handler.arg = NULL};
  ucp_ep_create(worker, &ep_params, &ep);

  // Register AM handler to receive rkey from server
  ucp_am_handler_param_t am_param = {.field_mask =
                                         UCP_AM_HANDLER_PARAM_FIELD_ID |
                                         UCP_AM_HANDLER_PARAM_FIELD_CB |
                                         UCP_AM_HANDLER_PARAM_FIELD_ARG,
                                     .id = AM_ID,
                                     .cb = rkey_recv_cb,
                                     .arg = ep};
  ucp_worker_set_am_recv_handler(worker, &am_param);

  // Wait until rkey is received
  while (!rkey_received) {
    ucp_worker_progress(worker);
    usleep(100);
  }

  for (int i = 0; i < 1; i++) {
    // Send Active Message
    const char *msg = "Hello, UCX!";
    ucp_request_param_t am_param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                                    UCP_OP_ATTR_FIELD_USER_DATA,
                                    .cb.send = send_cb,
                                    .user_data = (void *)(uintptr_t)i};
    void *am_req =
        ucp_am_send_nbx(ep, AM_ID, NULL, 0, msg, strlen(msg) + 1, &am_param);
    if (UCS_PTR_IS_PTR(am_req)) {
      while (ucp_request_check_status(am_req) == UCS_INPROGRESS)
        ucp_worker_progress(worker);
      ucp_request_free(am_req);
    } else if (!UCS_STATUS_IS_ERR((ucs_status_t)(uintptr_t)am_req)) {
      send_cb(NULL, (ucs_status_t)(uintptr_t)am_req, am_param.user_data);
    } else {
      fprintf(stderr, "Request failed: %s\n",
              ucs_status_string((ucs_status_t)(uintptr_t)am_req));
    }

    // Send RDMA PUT
    char *rdma_data = malloc(RDMA_MSG_SIZE);
    snprintf(rdma_data, RDMA_MSG_SIZE, "%s %d", RDMA_MSG, i);
    fprintf(stderr, "Client: Sending RDMA data: %s\n", rdma_data);

    ucp_request_param_t put_param = {.op_attr_mask =
                                         UCP_OP_ATTR_FIELD_CALLBACK |
                                         UCP_OP_ATTR_FIELD_USER_DATA,
                                     .cb.send = rdma_cb,
                                     .user_data = (void *)(uintptr_t)i};

    void *put_req = ucp_put_nbx(ep, rdma_data, RDMA_MSG_SIZE,
                                global_remote_addr, global_rkey, &put_param);
    if (UCS_PTR_IS_PTR(put_req)) {
      while (ucp_request_check_status(put_req) == UCS_INPROGRESS)
        ucp_worker_progress(worker);
      ucp_request_free(put_req);
    } else if (!UCS_STATUS_IS_ERR((ucs_status_t)(uintptr_t)put_req)) {
      rdma_cb(NULL, (ucs_status_t)(uintptr_t)put_req, put_param.user_data);
    } else {
      fprintf(stderr, "Request failed: %s\n",
              ucs_status_string((ucs_status_t)(uintptr_t)put_req));
    }
    fprintf(stderr, "Client: RDMA data sent\n");

    free(rdma_data);
    usleep(500000);
  }

  sleep(1);
  if (global_rkey != NULL) {
    ucp_rkey_destroy(global_rkey);
  }
  ucp_ep_destroy(ep);
  ucp_worker_destroy(worker);
  ucp_cleanup(context);
  return 0;
}

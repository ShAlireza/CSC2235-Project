#include "ucx_rdma_client.h"
#include "ucp/api/ucp.h"
#include "ucp/api/ucp_compat.h"
#include <cstring>
#include <iostream>
#include <unistd.h>

#define AM_ID 1
#define PORT 13337

static void send_cb(void *request, ucs_status_t status, void *user_data) {
  int id = (int)(uintptr_t)user_data;
  std::cout << "Client: AM message " << id
            << " sent (status = " << ucs_status_string(status) << ")\n";
}

static void rdma_cb(void *request, ucs_status_t status, void *user_data) {
  int id = (int)(uintptr_t)user_data;
  std::cout << "Client: RDMA write " << id
            << " completed (status = " << ucs_status_string(status) << ")\n";
}

static ucs_status_t rkey_recv_cb(void *arg, const void *header,
                                 size_t header_length, void *data,
                                 size_t length,
                                 const ucp_am_recv_param_t *param) {
  auto *client = static_cast<UcxRdmaClient *>(arg);
  if (length == sizeof(uint64_t)) {
    client->remote_addr = *((uint64_t *)data);
  } else {
    ucs_status_t status = ucp_ep_rkey_unpack(client->ep, data, &client->rkey);
    if (status == UCS_OK) {
      client->rkey_received = true;
    }
  }
  return UCS_OK;
}

UcxRdmaClient::UcxRdmaClient(const std::string &server_ip, size_t buffer_size,
                             size_t chunk_size)
    : buffer_size(buffer_size), chunk_size(chunk_size) {
  init_ucx(server_ip);
  send_metadata();
  wait_for_rkey();
  start_sender_thread();
}

UcxRdmaClient::~UcxRdmaClient() {
  finish();
  sender_thread.join();
  if (rkey)
    ucp_rkey_destroy(rkey);
  ucp_ep_destroy(ep);
  ucp_worker_destroy(worker);
  ucp_cleanup(context);
}

/*
Does init_worker,
registers the AM handler to receive the rkey and remote_addr from the server
Creates the endpoint to the server
*/
void UcxRdmaClient::init_ucx(const std::string &server_ip) {
  ucp_params_t params = {.field_mask = UCP_PARAM_FIELD_FEATURES,
                         .features = UCP_FEATURE_AM | UCP_FEATURE_RMA};
  ucp_init(&params, nullptr, &context);

  ucp_worker_params_t worker_params = {};
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  ucp_worker_create(context, &worker_params, &worker);

  // AM handler for rkey + remote_addr
  ucp_am_handler_param_t am_param = {.field_mask =
                                         UCP_AM_HANDLER_PARAM_FIELD_ID |
                                         UCP_AM_HANDLER_PARAM_FIELD_CB |
                                         UCP_AM_HANDLER_PARAM_FIELD_ARG,
                                     .id = AM_ID,
                                     .cb = rkey_recv_cb,
                                     .arg = this};
  ucp_worker_set_am_recv_handler(worker, &am_param);

  // Create endpoint
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(PORT);
  inet_pton(AF_INET, server_ip.c_str(), &addr.sin_addr);

  ucp_ep_params_t ep_params = {};
  ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS |
                         UCP_EP_PARAM_FIELD_SOCK_ADDR |
                         UCP_EP_PARAM_FIELD_ERR_HANDLER;
  ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
  ep_params.sockaddr.addr = (sockaddr *)&addr;
  ep_params.sockaddr.addrlen = sizeof(addr);
  ep_params.err_handler.cb = nullptr;
  ep_params.err_handler.arg = nullptr;
  ucp_ep_create(worker, &ep_params, &ep);
}

void UcxRdmaClient::send_metadata() {
  std::string msg =
      std::to_string(chunk_size) + " " + std::to_string(buffer_size);
  ucp_request_param_t param = {};
  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  param.user_data = (void *)(uintptr_t)0;
  param.cb.send = send_cb;
  void *req = ucp_am_send_nbx(ep, AM_ID, nullptr, 0, msg.c_str(),
                              msg.size() + 1, &param);
  if (UCS_PTR_IS_PTR(req)) {
    while (ucp_request_check_status(req) == UCS_INPROGRESS)
      ucp_worker_progress(worker);
    ucp_request_free(req);
  }
}

void UcxRdmaClient::wait_for_rkey() {
  while (!rkey_received)
    ucp_worker_progress(worker);
}

void UcxRdmaClient::send_chunk(int *data, size_t size) {
  if (!this->first_chunk_started) {
    this->first_chunk_started = true;
    // Print timestamp in nanoseconds
    auto now = std::chrono::high_resolution_clock::now();
    auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch())
          .count();
    std::cout << "RDMA Client: First chunk started at timestamp " << duration
            << "\n";
  }
  ucp_request_param_t put_param = {};
  put_param.op_attr_mask =
      UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  put_param.user_data = (void *)(uintptr_t)(requests.size());
  put_param.cb.send = rdma_cb;
  void *req =
      ucp_put_nbx(ep, data, size, remote_addr + current_offset + sizeof(int),
                  rkey, &put_param); // sizeof(int) is because we are using
                                     // first index as a counter
  ucp_request_param_t counter_param = {};
  counter_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                               UCP_OP_ATTR_FIELD_USER_DATA;
  counter_param.user_data = (void *)(uintptr_t)(requests.size());
  counter_param.cb.send = rdma_cb;

  // std::cout << "[RDMA] Writing chunk at offset " << current_offset
  //           << " | First value: " << data[0] << std::endl;
  current_offset += size;

  int *counter_value = new int[1];
  *counter_value = current_offset / sizeof(int);
  void *counter_req = ucp_put_nbx(ep, counter_value, sizeof(int), remote_addr,
                                  rkey, &counter_param);

  // print the chunk data
  std::cout << "Client: Sending chunk of size " << size << " at address "
            << remote_addr << "\n";
  // for (size_t i = 0; i < size / sizeof(int); ++i) {
  //     std::cout << data[i] << " ";
  // }
  // std::cout << "\n";

  std::cout << "Counter value was " << *counter_value << "\n";


  std::lock_guard<std::mutex> lock(requests_mutex);
  requests.push(req);
  requests.push(counter_req);
}

void UcxRdmaClient::send_finish() {
  // Send a finish message to the server
  ucp_request_param_t put_param = {};
  put_param.op_attr_mask =
      UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  put_param.user_data = (void *)(uintptr_t)(requests.size());
  put_param.cb.send = rdma_cb;
  int *data = new int[1];
  *data = -1;
  void *req = ucp_put_nbx(ep, data, sizeof(int), remote_addr, rkey,
                          &put_param); // we are using
                                       // first index as a counter

  requests.push(req);
}

void UcxRdmaClient::start_sender_thread() {
  sender_thread = std::thread(&UcxRdmaClient::sender_loop, this);
  sender_thread.detach();
}

void UcxRdmaClient::sender_loop() {
  while (!finished || !requests.empty()) {
    void *req = nullptr;
    {
      std::unique_lock<std::mutex> lock(requests_mutex);
      if (!requests.empty()) {
        req = requests.front();
        requests.pop();
      }
      lock.unlock();
    }

    if (req) {
      if (UCS_PTR_IS_PTR(req)) {
        while (ucp_request_check_status(req) == UCS_INPROGRESS)
          ucp_worker_progress(worker);
        ucp_request_free(req);
      } else if (!UCS_STATUS_IS_ERR((ucs_status_t)(uintptr_t)req)) {
        rdma_cb(nullptr, (ucs_status_t)(uintptr_t)req, nullptr);
      }
    }
    ucp_worker_progress(worker);
    usleep(1000);
  }

  // std::cout << "UcxRdmaClient: Sender loop flushed.\n";
  // Print timestamp in nanoseconds
  auto now = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch())
          .count();
  std::cout << "RDMA Client: Sender loop finished at timestamp " << duration
            << "\n";
  this->done_flushing = true;
}

void UcxRdmaClient::finish() { finished = true; }

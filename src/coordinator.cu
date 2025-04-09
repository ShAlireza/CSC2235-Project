#include "coordinator.h"
#include "logging.h"
#include <arpa/inet.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

void handler(int socket, void *data) {
  Coordinator *coord = (Coordinator *)data;

  WorkerInfo *worker_info = (WorkerInfo *)malloc(sizeof(WorkerInfo));

  // Receives worker addr size
  recv(socket, &worker_info->address_size, sizeof(int), 0);

  close(socket);
}

int run_listener(Coordinator *coord, connection_handler_t handler) {
  ucp_worker_get_address(coord->worker, &coord->worker_address,
                         &coord->worker_address_size);


  int server_fd, *new_socket;

  struct sockaddr_in address;
  int addrlen = sizeof(address);

  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    LOG_ERR("socket failed");
    return -1;
  }

  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY; // WARN: Listening on 0.0.0.0
  address.sin_port = htons(coord->port);

  if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
    LOG_ERR("bind failed");
    return -1;
  }

  if (listen(server_fd, 10) < 0) {
    LOG_ERR("listen failed");
    return -1;
  }

  LOG_INFO("Listening on port %d", coord->port);

  while (1) {
    new_socket = (int *)malloc(sizeof(int));
    if ((*new_socket = accept(server_fd, (struct sockaddr *)&address,
                              (socklen_t *)&addrlen)) < 0) {
      LOG_ERR("accept failed");
      return -1;
    }
    LOG_INFO("New connection accepted");

    std::thread t(handler, new_socket, (void *)coord);
  }

  return 0;
}

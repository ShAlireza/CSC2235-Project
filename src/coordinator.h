#ifndef COORDINTOR_H
#define COORDINATOR_H

#include "ucp/api/ucp_compat.h"
#include <cstdint>
#include <thread>
#include <ucp/api/ucp.h>

#define MAX_WORKERS 100

typedef struct {
  int address_size;
  void *worker_addr;
  ucp_address_t *ucp_addr;
  bool is_sender;
} WorkerInfo;

typedef struct {
  int port;
  int af;
  WorkerInfo workers[MAX_WORKERS];
  int num_workers;
} Coordinator;

typedef void (*connection_handler_t)(int, void *);

int run_listener(Coordinator *coord, connection_handler_t handler);

#endif // COORDINATOR_H

#include <ucp/api/ucp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <unistd.h>

#define AM_ID 1
#define PORT 13337
#define RDMA_BUFFER_SIZE 128

typedef struct {
    uint64_t remote_addr;
    // rkey will follow this header
} __attribute__((packed)) rdma_info_t;

static ucp_context_h context;
static ucp_worker_h worker;
static ucp_mem_h memh;
static void *rdma_buffer = NULL;

static int init_worker(ucp_context_h ucp_context, ucp_worker_h *ucp_worker)
{
    ucp_worker_params_t worker_params;
    ucs_status_t status;
    int ret = 0;

    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    status = ucp_worker_create(ucp_context, &worker_params, ucp_worker);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_worker_create (%s)\n", ucs_status_string(status));
        ret = -1;
    }
    return ret;
}

ucs_status_t am_recv_cb(void *arg, const void *header, size_t header_length,
                        void *data, size_t length, const ucp_am_recv_param_t *param) {
    printf("Server received AM: %.*s\n", (int)length, (char*)data);
    return UCS_OK;
}

void send_rkey_to_client(ucp_ep_h ep) {
    void *rkey_buffer = NULL;
    size_t rkey_size = 0;
    ucs_status_t status = ucp_rkey_pack(context, memh, &rkey_buffer, &rkey_size);
    if (status != UCS_OK) {
        fprintf(stderr, "ucp_rkey_pack failed: %s\n", ucs_status_string(status));
        return;
    }

    size_t msg_size = sizeof(rdma_info_t) + rkey_size;
    char *msg = malloc(msg_size);
    rdma_info_t *info = (rdma_info_t*)msg;
    info->remote_addr = (uintptr_t)rdma_buffer;
    memcpy(msg + sizeof(rdma_info_t), rkey_buffer, rkey_size);

    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK,
        .cb.send = NULL
    };

    void *req = ucp_am_send_nbx(ep, AM_ID, NULL, 0, msg, msg_size, &param);
    if (UCS_PTR_IS_PTR(req)) {
        while (ucp_request_check_status(req) == UCS_INPROGRESS)
            ucp_worker_progress(worker);
        ucp_request_free(req);
    }

    ucp_rkey_buffer_release(rkey_buffer);
    free(msg);
    printf("Server: Sent rkey and remote_addr to client\n");
}

void on_connection(ucp_conn_request_h conn_request, void *arg) {
    ucp_worker_h worker = (ucp_worker_h)arg;
    ucp_ep_params_t ep_params = {
        .field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST,
        .conn_request = conn_request
    };
    ucp_ep_h ep;
    ucp_ep_create(worker, &ep_params, &ep);
    printf("Server: client connected\n");
    send_rkey_to_client(ep);
}

int main() {
    ucp_listener_h listener;
    ucs_status_t status;

    // Init context
    ucp_params_t params = {
        .field_mask = UCP_PARAM_FIELD_FEATURES,
        .features = UCP_FEATURE_AM | UCP_FEATURE_RMA
    };
    status = ucp_init(&params, NULL, &context);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_init (%s)\n", ucs_status_string(status));
        return -1;
    }
    printf("Context initialized\n");

    if (init_worker(context, &worker) != 0) {
        fprintf(stderr, "failed to init_worker\n");
        ucp_cleanup(context);
        return -1;
    }
    printf("Worker initialized\n");

    // Allocate and register buffer for RDMA
    rdma_buffer = calloc(1, RDMA_BUFFER_SIZE);
    ucp_mem_map_params_t mmap_params = {
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                      UCP_MEM_MAP_PARAM_FIELD_LENGTH,
        .address = rdma_buffer,
        .length  = RDMA_BUFFER_SIZE
    };
    status = ucp_mem_map(context, &mmap_params, &memh);
    if (status != UCS_OK) {
        fprintf(stderr, "ucp_mem_map failed: %s\n", ucs_status_string(status));
        return -1;
    }
    printf("Server RDMA buffer registered at %p\n", rdma_buffer);

    // Set AM callback
    ucp_am_handler_param_t am_param = {
        .field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                      UCP_AM_HANDLER_PARAM_FIELD_CB,
        .id = AM_ID,
        .cb = am_recv_cb
    };
    ucp_worker_set_am_recv_handler(worker, &am_param);

    // Listener
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(PORT),
        .sin_addr.s_addr = INADDR_ANY
    };
    ucp_listener_params_t listener_params = {
        .field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                      UCP_LISTENER_PARAM_FIELD_CONN_HANDLER,
        .sockaddr = { .addr = (struct sockaddr*)&addr, .addrlen = sizeof(addr) },
        .conn_handler = { .cb = on_connection, .arg = worker }
    };
    ucp_listener_create(worker, &listener_params, &listener);
    printf("Server is listening on port %d\n", PORT);

    while (1) {
        ucp_worker_progress(worker);
        // print strlen of rdma_buffer
        //printf("strlen(rdma_buffer) = %ld\n", strlen((char*)rdma_buffer));
        if (((char*)rdma_buffer)[0] != '\0') {
            printf("[RDMA] Server received data: %s\n", (char*)rdma_buffer);
            memset(rdma_buffer, 0, RDMA_BUFFER_SIZE);
        }
        usleep(1000);
    }

    ucp_mem_unmap(context, memh);
    free(rdma_buffer);
    ucp_listener_destroy(listener);
    ucp_worker_destroy(worker);
    ucp_cleanup(context);
    return 0;
}

#include <ucp/api/ucp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <unistd.h>

#define AM_ID 1
#define PORT 13337

//volatile int received = 0;

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
    printf("Server received: %.*s\n", (int)length, (char*)data);
    //received = 1;
    return UCS_OK;
}

void on_connection(ucp_conn_request_h conn_request, void *arg) {
    ucp_worker_h worker = (ucp_worker_h)arg;
    ucp_ep_params_t ep_params = {
        .field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST,
        .conn_request = conn_request
    };
    ucp_ep_h ep;
    ucp_ep_create(arg, &ep_params, &ep);
}

int main() {
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_listener_h listener;
    ucs_status_t status;

    // Init context
    ucp_params_t params = {
        .field_mask = UCP_PARAM_FIELD_FEATURES,
        .name = "ucx_server",
        .features = UCP_FEATURE_AM
    };
    status = ucp_init(&params, NULL, &context);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_init (%s)\n", ucs_status_string(status));
        return -1;
    }

    // Print to check if context is initialized
    printf("Context initialized\n");

    ucp_worker_params_t worker_params = {
        .field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCS_THREAD_MODE_SINGLE
    };

    memset(&worker_params, 0, sizeof(worker_params));

    // Init worker using init_worker function
    if (init_worker(context, &worker) != 0) {
        fprintf(stderr, "failed to init_worker\n");
        ucp_cleanup(context);
        return -1;
    }
    

    // Print to check if worker is initialized
    printf("Worker initialized\n");

    // Set AM callback
    ucp_am_handler_param_t am_param = {
        .field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_CB,
        .id = AM_ID,
        .cb = am_recv_cb
    };
    ucp_worker_set_am_recv_handler(worker, &am_param);

    // Set up listener
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
        usleep(1000); // 1ms
    }

    ucp_listener_destroy(listener);
    ucp_worker_destroy(worker);
    ucp_cleanup(context);
    return 0;
}

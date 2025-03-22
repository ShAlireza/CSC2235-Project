#include <ucp/api/ucp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <unistd.h>

#define AM_ID 1
#define PORT 13337


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

void send_cb(void *request, ucs_status_t status, void *user_data) {
    int id = (int)(uintptr_t)user_data;
    printf("Client: message %d sent successfully (status = %s)\n", id, ucs_status_string(status));
    if (request != NULL) {
        ucp_request_free(request);
    }
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
    ucp_params_t params = {
        .field_mask = UCP_PARAM_FIELD_FEATURES,
        .features = UCP_FEATURE_AM
    };
    ucp_init(&params, NULL, &context);

    // Init worker using init_worker function
    if (init_worker(context, &worker) != 0) {
        fprintf(stderr, "failed to init_worker\n");
        ucp_cleanup(context);
        return -1;
    }
    

    // Create endpoint
    struct sockaddr_in server_addr = {
        .sin_family = AF_INET,
        .sin_port = htons(PORT)
    };
    inet_pton(AF_INET, server_ip, &server_addr.sin_addr);

    ucp_ep_params_t ep_params = {
        .field_mask = UCP_EP_PARAM_FIELD_FLAGS |
                      UCP_EP_PARAM_FIELD_SOCK_ADDR,
        .flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER,
        .sockaddr = { .addr = (struct sockaddr*)&server_addr, .addrlen = sizeof(server_addr) }
    };

    ucp_ep_create(worker, &ep_params, &ep);

    // Send message
    const char *msg = "Hello, UCX!";

    // send 5 requests
    for (int i = 0; i < 1000; i++) {
        ucp_request_param_t param = {
            .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                            UCP_OP_ATTR_FIELD_USER_DATA,
            .cb.send = send_cb,
            .user_data = (void*)(uintptr_t)i  // Pass index as user_data if you want
        };

        void *request = ucp_am_send_nbx(ep, AM_ID, NULL, 0, msg, strlen(msg) + 1, &param);
        if (UCS_PTR_IS_PTR(request)) {
            // wait for completion
            while (ucp_request_check_status(request) == UCS_INPROGRESS)
                ucp_worker_progress(worker);
        } else if (!UCS_STATUS_IS_ERR((ucs_status_t)(uintptr_t)request)) {
            // If completed immediately, manually call send_cb
            send_cb(NULL, (ucs_status_t)(uintptr_t)request, param.user_data);
        } else {
            fprintf(stderr, "failed to send message (%s)\n", ucs_status_string((ucs_status_t)(uintptr_t)request));
        }
        
    }    

    // Cleanup
    sleep(1); // wait for server to receive
    ucp_ep_destroy(ep);
    ucp_worker_destroy(worker);
    ucp_cleanup(context);
    return 0;
}

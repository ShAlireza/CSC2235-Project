#pragma once

#include <ucp/api/ucp.h>
#include <arpa/inet.h>
#include <queue>
#include <string>
#include <thread>
#include <mutex>

class UcxRdmaClient {
public:
    UcxRdmaClient(const std::string &server_ip, size_t buffer_size, size_t chunk_size);
    ~UcxRdmaClient();

    void send_chunk(int *data, size_t size);
    void send_finish();
    void finish(); // Signals the sender thread to flush and exit

    void init_ucx(const std::string &server_ip);
    void start_sender_thread();
    void sender_loop();

    void send_metadata();
    void wait_for_rkey();

    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;
    ucp_rkey_h rkey;
    uint64_t remote_addr;

    std::queue<void *> requests;
    std::thread sender_thread;
    std::mutex requests_mutex;
    size_t current_offset = 0;

    bool rkey_received = false;
    bool finished = false;

    size_t buffer_size;
    size_t chunk_size;
};

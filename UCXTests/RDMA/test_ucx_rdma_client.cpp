#include "ucx_rdma_client.h"
#include <cstdlib>
#include <iostream>

#define DEFAULT_CHUNK_SIZE 64
#define DEFAULT_BUFFER_SIZE 512

int *generate_random_data(size_t buffer_size) {
    int *data = (int *)malloc(buffer_size * sizeof(int));
    if (!data) {
        std::cerr << "Failed to allocate buffer\n";
        return nullptr;
    }

    for (size_t i = 0; i < buffer_size; ++i) {
        data[i] = rand() % 1000;
    }

    return data;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <server_ip>\n";
        return 1;
    }

    std::string server_ip = argv[1];
    size_t chunk_size = DEFAULT_CHUNK_SIZE;
    size_t buffer_size = DEFAULT_BUFFER_SIZE;

    std::cout << "Initializing RDMA client\n";
    UcxRdmaClient client(server_ip, buffer_size * sizeof(int), chunk_size * sizeof(int));

    int *buffer = generate_random_data(buffer_size);
    if (!buffer) return 1;

    std::cout << "Sending buffer of size " << buffer_size << " in chunks of " << chunk_size << "\n";
    for (size_t offset = 0; offset < buffer_size; offset += chunk_size) {
        size_t remaining = std::min(chunk_size, buffer_size - offset);
        client.send_chunk(buffer + offset, remaining * sizeof(int));
    }

    client.finish();

    free(buffer);
    std::cout << "Done.\n";
    return 0;
}
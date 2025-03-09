#include <chrono>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <utils.h>

/*
This will be a worker function for both nodes that will be communicating via RDMA. There will be a boolean to determine whether this node is sending or recieving data via RDMA
*/

// Create a worker class with C++

typedef struct {
    bool is_sender;
    int num_threads;
    int gpu_id;
    void *data;
    void *staged_data;
    int size;
    // need to create buffer to store ids of other workers
    void ** workers;
    int num_workers;

    void (*on_path)(void* chunk, int chunk_size);
} Worker;


// Initialize worker
Worker* init_worker(bool is_sender, int num_threads, int gpu_id, void *data, void* staged_data, int size) {
    Worker* worker = (Worker*)malloc(sizeof(Worker));
    worker->is_sender = is_sender;
    worker->num_threads = num_threads;
    worker->gpu_id = gpu_id;
    worker->data = data;
    worker->staged_data = staged_data;
    worker->size = size;
    return worker;
}

// Free worker
void free_worker(Worker* worker) {
    free(worker);
}


// create function to either send or recieve data based on the boolean
void worker_run(Worker* worker) {
    std::thread threads[worker->num_threads];
    for (int i = 0; i < worker->num_threads; i++) {
        threads[i] = std::thread(process, worker, i);
    }
    for (int i = 0; i < worker->num_threads; i++) {
        threads[i].join();
    }



    if (worker->is_sender) {
        printf("I am the sender\n");
        
    } else {
        printf("I am the reciever\n");
    }
}



// create a function to process data, split the data from ITEMS_COUNT macro, and send or recieve data based on the boolean
void process(Worker* worker, int thread_id) {
    // Find the chunk size
    cudaSetDevice(worker->gpu_id);
    int chunk_size = worker->size / worker->num_threads;
    // If sender, recieve data from GPU

    if (worker->is_sender) {
        // memcopy data from the gpu to the host
        cudaMemcpy(worker->staged_data + thread_id * chunk_size, worker->data + thread_id * chunk_size, chunk_size, cudaMemcpyDeviceToHost);
        // Run onpath function, which reduces the data
        worker->on_path(worker->staged_data + thread_id * chunk_size, chunk_size);
        // Send data to the other node
        send(worker, thread_id, chunk_size, worker->staged_data + thread_id * chunk_size);
    }
    
    if (!worker->is_sender) {
        printf("Recieving data from thread %d\n", thread_id);
    }
}

// crate send function
void send(Worker* worker, int thread_id, int chunk_size, void* data) {
    printf("Sending data from thread %d\n", thread_id);
}

// onPath function
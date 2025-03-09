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

    // On_path functions return the result buffer size
    void (*on_path)(void* chunk, int chunk_size, void ** result_buffer, int* result_buffer_size);
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
        void *result_buffer;
        int result_buffer_size;
        worker->on_path(worker->staged_data + thread_id * chunk_size, chunk_size, &result_buffer, &result_buffer_size);
        // Send data to the other node
        send(worker, thread_id, &result_buffer_size, &result_buffer);
    }
    
    if (!worker->is_sender) {
        printf("Recieving data from thread %d\n", thread_id);
    }
}

// crate send function
void send(Worker* worker, int thread_id, int *result_buffer_size, void** result_buffer) {
    // We are going to send the data to the other node using RDMA
    // We first need to send the result buffer size, so the other node knows how much data to expect


    // We then send the result buffer

}

// onPath function




// Create main function
/*
Main should first generate the data, then create the worker, then run the worker
*/
int main() {
    int num_gpus = 2;
    // array of pointers to buffers
    int * host_buffers[num_gpus];
    int * gpu_buffers[num_gpus];
    int * src_gpu_ids[num_gpus];
    // for each gpu, create host buffer and gpu buffer, then generate data
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        int * host_buffer;
        cudaMallocHost((void **)&host_buffer, ITEMS_COUNT * sizeof(int));
        int * gpu_buffer;
        cudaMalloc((void **)&gpu_buffer, ITEMS_COUNT * sizeof(int));
        host_buffers[i] = host_buffer;
        gpu_buffers[i] = gpu_buffer;
        src_gpu_ids[i] = i;

        // Generate data (in utils.h)
        generate_data(i, host_buffer, gpu_buffer, ITEMS_COUNT * sizeof(int), i);
    }

    // We now need to create a worker for each gpu
    Worker* workers[num_gpus];
    // we cant loop here because we need to tell each worker if they are a sender or reciever
    workers[0] = init_worker(true, 4, 0, gpu_buffers[0], host_buffers[0], ITEMS_COUNT * sizeof(int));
    workers[1] = init_worker(false, 4, 1, gpu_buffers[1], host_buffers[1], ITEMS_COUNT * sizeof(int));


    // Run workers
    worker_run(workers[0]);
    worker_run(workers[1]);

    // Free workers
    free_worker(workers[0]);
    free_worker(workers[1]);
    return 0;
}
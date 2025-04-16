# UCX installation paths
UCX_INCLUDE = -I$(UCX_PATH)/include
UCX_LIB = -L$(UCX_PATH)/lib -lucp -lucs

# Compiler settings
CC = nvcc
CFLAGS = $(UCX_INCLUDE)
LDFLAGS = $(UCX_LIB)

# Targets
all: ucx_server ucx_rdma_server ucx_rdma_client test_ucx_rdma_client

ucx_server: ucx_server.cpp
	$(CC) -o ucx_server ucx_server.cpp $(CFLAGS) $(LDFLAGS)

ucx_rdma_server: ucx_rdma_server.cu
	$(CC) -o ucx_rdma_server ucx_rdma_server.cpp $(CFLAGS) $(LDFLAGS)

ucx_rdma_client: ucx_rdma_client.cpp ucx_rdma_client.h
	$(CC) -c ucx_rdma_client.cpp $(CFLAGS)

test_ucx_rdma_client: test_ucx_rdma_client.cpp ucx_rdma_client.o
	$(CC) -o test_ucx_rdma_client test_ucx_rdma_client.cpp ucx_rdma_client.o $(CFLAGS) $(LDFLAGS)

clean:
	rm -f ucx_server ucx_rdma_server ucx_rdma_client.o test_ucx_rdma_client

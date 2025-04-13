# UCX installation paths
UCX_PATH = /home/tabbahch/Desktop/ucx-1.17.0/install
UCX_INCLUDE = -I$(UCX_PATH)/include
UCX_LIB = -L$(UCX_PATH)/lib -lucp -lucs

# Compiler settings
CC = g++
CFLAGS = $(UCX_INCLUDE) -pthread
LDFLAGS = $(UCX_LIB)

# Targets
all: ucx_server ucx_rdma_client test_ucx_rdma_client

ucx_server: ucx_server.cpp
	$(CC) -o ucx_server ucx_server.cpp $(CFLAGS) $(LDFLAGS)

ucx_rdma_client: ucx_rdma_client.cpp ucx_rdma_client.h
	$(CC) -c ucx_rdma_client.cpp $(CFLAGS)

test_ucx_rdma_client: test_ucx_rdma_client.cpp ucx_rdma_client.o
	$(CC) -o test_ucx_rdma_client test_ucx_rdma_client.cpp ucx_rdma_client.o $(CFLAGS) $(LDFLAGS)

clean:
	rm -f ucx_server ucx_rdma_client.o test_ucx_rdma_client

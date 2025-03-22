# UCX installation paths
UCX_PATH = /home/tabbahch/Desktop/UCXTests/ucx-1.17.0/install
UCX_INCLUDE = -I$(UCX_PATH)/include
UCX_LIB = -L$(UCX_PATH)/lib -lucp -lucs

# Compiler settings
CC = gcc
CFLAGS = $(UCX_INCLUDE) $(UCX_LIB)

# Targets
all: ucx_server ucx_client

ucx_server: ucx_server.c
	$(CC) -o ucx_server ucx_server.c $(CFLAGS)

ucx_client: ucx_client.c
	$(CC) -o ucx_client ucx_client.c $(CFLAGS)

clean:
	rm -f ucx_server ucx_client

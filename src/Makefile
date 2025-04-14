# CUDA and C++ compiler settings
NVCC = nvcc
CXX = g++
UCX_PATH = /home/tabbahch/Desktop/ucx-1.17.0/install

UCX_INCLUDE = -I$(UCX_PATH)/include
UCX_LIB = -L$(UCX_PATH)/lib -lucp -lucs
CXXFLAGS = -std=c++17 -pthread $(UCX_INCLUDE)
LDFLAGS = $(UCX_LIB)

# Paths to RDMA source files
RDMA_DIR = RDMA
RDMA_OBJS = $(RDMA_DIR)/ucx_rdma_client.o

# Main files
SRC_OBJS = distinct_merge.o
CU_OBJ = deduplicate.o

# Final binary
TARGET = deduplicate

all: $(TARGET)

$(RDMA_DIR)/ucx_rdma_client.o: $(RDMA_DIR)/ucx_rdma_client.cpp $(RDMA_DIR)/ucx_rdma_client.h
	$(CXX) -c $< -o $@ $(CXXFLAGS)

distinct_merge.o: distinct_merge.cu distinct_merge.h
	$(NVCC) -c $< -o $@ -Xcompiler "-std=c++17 -pthread" $(UCX_INCLUDE)

deduplicate.o: deduplicate.cu distinct_merge.h
	$(NVCC) -c $< -o $@ -Xcompiler "-std=c++17 -pthread" $(UCX_INCLUDE)

$(TARGET): $(RDMA_OBJS) $(SRC_OBJS) $(CU_OBJ)
	$(NVCC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o $(RDMA_DIR)/*.o $(TARGET)
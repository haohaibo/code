CUDA_PATH ?= /dev/shm/cuda-11.4
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

LIBRARIES :=

LIBRARIES += -lcublas

# Gencode arguments
SMS ?= 35 37 50 52 60 61 70 75 80 86

ifeq ($(GENCODE_FLAGS),)
# generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS), $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

all: build

build: gemm_naive

EXEC ?= @echo

gemm_naive.o: gemm_naive.cu
	$(NVCC) $(GENCODE_FLAGS) -o $@ -c $<

gemm_naive: gemm_naive.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

clean:
	rm -rfv gemm_naive gemm_naive.o


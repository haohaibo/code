CUDA_PATH ?= /data/cuda-12.6
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

LIBRARIES :=

LIBRARIES += -lcublas

# Gencode arguments
SMS ?= 50 52 60 61 70 75 80 86 89

ifeq ($(GENCODE_FLAGS),)
# generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS), $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

all: build

build: gemm_naive gemm_coalescing gemm_shared_mem gemm_shared_mem_1d_tiling gemm_shared_mem_1d_tiling_As_conti \
		gemm_shared_mem_2d_block_tiling gemm_shared_mem_2d_block_tiling_v0 gemm_shared_mem_2d_block_tiling_as_lds128 \
		gemm_shared_mem_2d_block_tiling_as_lds128_vec_gmem

BUILD_ENABLED := 1

ifeq ($(BUILD_ENABLED), 0)
EXEC ?= @echo "[@]"
endif

gemm_naive.o: gemm_naive.cu
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ -c $<

gemm_naive: gemm_naive.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

gemm_coalescing.o: gemm_coalescing.cu
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ -c $<

gemm_coalescing: gemm_coalescing.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

gemm_shared_mem.o: gemm_shared_mem.cu
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ -c $<

gemm_shared_mem: gemm_shared_mem.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)	

gemm_shared_mem_1d_tiling.o: gemm_shared_mem_1d_tiling.cu
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ -c $<

gemm_shared_mem_1d_tiling: gemm_shared_mem_1d_tiling.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

gemm_shared_mem_1d_tiling_As_conti.o: gemm_shared_mem_1d_tiling_As_conti.cu
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ -c $<

gemm_shared_mem_1d_tiling_As_conti: gemm_shared_mem_1d_tiling_As_conti.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

gemm_shared_mem_2d_block_tiling_v0.o: gemm_shared_mem_2d_block_tiling_v0.cu
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ -c $<

gemm_shared_mem_2d_block_tiling_v0: gemm_shared_mem_2d_block_tiling_v0.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

gemm_shared_mem_2d_block_tiling.o: gemm_shared_mem_2d_block_tiling.cu
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ -c $<

gemm_shared_mem_2d_block_tiling: gemm_shared_mem_2d_block_tiling.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

gemm_shared_mem_2d_block_tiling_as_lds128.o: gemm_shared_mem_2d_block_tiling_as_lds128.cu
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ -c $<

gemm_shared_mem_2d_block_tiling_as_lds128: gemm_shared_mem_2d_block_tiling_as_lds128.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

gemm_shared_mem_2d_block_tiling_as_lds128_vec_gmem.o: gemm_shared_mem_2d_block_tiling_as_lds128_vec_gmem.cu
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ -c $<

gemm_shared_mem_2d_block_tiling_as_lds128_vec_gmem: gemm_shared_mem_2d_block_tiling_as_lds128_vec_gmem.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

clean:
	rm -rfv gemm_naive gemm_naive.o gemm_coalescing.o gemm_coalescing gemm_shared_mem.o gemm_shared_mem \
		gemm_shared_mem_1d_tiling.o gemm_shared_mem_1d_tiling gemm_shared_mem_1d_tiling_As_conti \
		gemm_shared_mem_1d_tiling_As_conti.o gemm_shared_mem_2d_block_tiling.o \
		gemm_shared_mem_2d_block_tiling gemm_shared_mem_2d_block_tiling_v0 \
		gemm_shared_mem_2d_block_tiling_v0.o \
		gemm_shared_mem_2d_block_tiling_as_lds128 \
		gemm_shared_mem_2d_block_tiling_as_lds128.o \
		gemm_shared_mem_2d_block_tiling_as_lds128_vec_gmem


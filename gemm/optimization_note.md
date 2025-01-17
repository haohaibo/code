# 问题描述：

C = A * B

A是M\*K大小的矩阵，B是K\*N大小的矩阵

C是M\*N大小的矩阵

A矩阵按行优先存储，B矩阵按行优先存储

C矩阵也按行优先储存

Naive gemm

直观的实现，每个thread(threadIdx.x, threadIdx.y)计算一个C中元素

每个thread读取对应的一行A和对应的一列B，沿着K方向做乘加操作


# Optimization 1

## Gobal memory coalescing

让一个warp中的连续的线程访问连续的全局内存
全局访存合并，使用长字节load




### Occupancy计算


#### GeForce GT 730, Kepler架构

| Metric      | Value |
| ----------- | ----------- |
| Name      | NVIDIA GeForce GT 730|
| Compute capability | 3.5 |
| Multiprocessor count | 2 |
| Maximum number of threads per block | 1024 |
| Maximum number of threads per multiprocessor | 2048 |
| Maximum number of blocks per mutiprocessor | 16 |
| Warp size | 32 |
| Maximum registers per block | 65536 |
| Registers per multiprocessor | 65536 |
| Maximum shared memory per block | 49152Bytes = 48KB |
| Reserved shared memory per block(Shared memory reserved by CUDA driver per block in bytes)| 0Byte |
| Shared memory per multiprocessor| 49152Bytes = 48KB |
| Total global memory | 2097283072Bytes = 2GB |


#### GeForce RTX 4050 Laptop GPU, Ada架构

| Metric      | Value |
| ----------- | ----------- |
| Name      | NVIDIA GeForce RTX 4050 Laptop GPU|
| Compute capability | 8.9 |
| Multiprocessor count | 20 |
| Maximum number of threads per block | 1024 |
| Maximum number of threads per multiprocessor | 1536 |
| Maximum number of blocks per mutiprocessor | 24 |
| Warp size | 32 |
| Maximum registers per block | 65536 |
| Registers per multiprocessor | 65536 |
| Maximum shared memory per block | 49152Bytes = 48KB |
| Reserved shared memory per block(Shared memory reserved by CUDA driver per block in bytes)| 1024Byte |
| Shared memory per multiprocessor| 102400Bytes = 100KB |
| Total global memory | 6082789376Bytes = 6GB |




# Optimization 2

## Use shared memory

计算寄存器 per thread，
shared memory per thread block
的使用情况

### 在GeForce GT 730 Kepler GPU

使用nvcc编译器查看
```
nvcc -Xptxas -v
```
例如下面，每个thread使用了26个registers
每个thread block使用了8192 Bytes shared memory
```
$ nvcc -Xptxas -v gemm_shared_mem.cu -o gemm_shared_mem -gencode arch=compute_35,code=sm_35 -lcublas
nvcc warning : The 'compute_35', 'compute_37', 'compute_50', 'sm_35', 'sm_37' and 'sm_50' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z5sgemmILj32EEviiiffPKfS1_Pf' for 'sm_35'
ptxas info    : Function properties for _Z5sgemmILj32EEviiiffPKfS1_Pf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 26 registers, 8192 bytes smem, 368 bytes cmem[0]
```

### gemm_shared_mem kernel 在GeForce GT 730 Kepler GPU的资源使用情况:

|Metric|Value|
|------|-----|
| Registers per Thread | 26 |
| SMEM per Block | 8192 Bytes |
| Threads per Block | 1024 |

### Shared Memory:

8KB per Block. 48KB per SM / 8KB per Block = 6 Blocks upper limit

### Threads
1024 Threads per Block. Max 2048 Threads per SM / 1024 Threads per Block = 2 Blocks upper limit

### Registers

26 Registers per Threads. 32 Threads per Warp. 26 * 32 = 832 Registers per Warp (without rounding).
If warp register allocation granularity is 256. Then 
ceil(26*32/256) * 256 = 1024. Allocate 1024 Registers per Warp.
We have 1024 Threads per Block / 32 Threads per Warp = 32 Warps.
So allcate (1024 Register per Warp) * (32 Warps) = 32K Registers per Block.

64K Registers per SM / 32K Registers per Block = 2 Blocks upper limit

So this kernel is limited by the number of threads per block and the number of registers per block. We cannot load more than 2 blocks per SM, giving us the final occupancy of (32 * 2) active warps / 64 max active warps = 100 %

100 % occupancy is good, so this doesn't explain why our kernel runs so slow.


### 在GeForce RTX 4050 Laptop Ada GPU

使用nvcc编译器查看
```
nvcc -Xptxas -v
```
例如下面，每个thread使用了38个registers
每个thread block使用了8192 Bytes shared memory
```
$ nvcc -Xptxas -v gemm_shared_mem.cu -o gemm_shared_mem -gencode arch=compute_89,code=sm_89 -lcublas
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z5sgemmILj32EEviiiffPKfS1_Pf' for 'sm_89'
ptxas info    : Function properties for _Z5sgemmILj32EEviiiffPKfS1_Pf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 38 registers, used 1 barriers, 8192 bytes smem, 400 bytes cmem[0]

```

### gemm_shared_mem kernel 在GeForce RTX 4050 Laptop Ada GPU的资源使用情况:

|Metric|Value|
|------|-----|
| Registers per Thread | 38 |
| SMEM per Block | 8192 Bytes |
| Threads per Block | 1024 |

### Shared Memory:

8KB per Block + 1KB per CUDA Runtime Usage per Block = 9KB per BLOCK.

 100KB per SM / 9KB per Block = 11.1 => 11 Blocks upper limit

### Threads
1024 Threads per Block. Max 1536 Threads per SM / 1024 Threads per Block = 1.5 => 1 Blocks upper limit

### Registers

38 Registers per Threads. 32 Threads per Warp. 38 * 32 = 1216 Registers per Warp (without rounding).
If warp register allocation granularity is 256. Then 
ceil(38*32/256) * 256 = 1280. Allocate 1280 Registers per Warp.
We have 1024 Threads per Block / 32 Threads per Warp = 32 Warps.
So allcate (1280 Register per Warp) * (32 Warps) = 40K Registers per Block.

64K Registers per SM / 40K Registers per Block = 1.6 => 1 Blocks upper limit

So this kernel is limited by the number of threads per block and the number of registers per block. We cannot load more than 1 blocks per SM, giving us the final occupancy of 32 active warps / 48 max active warps = 66 %

66 % occupancy is not too bad, so this doesn't explain why our kernel runs so slow.


It is more efficient to calculate a square of results per thread than a column of results because
we can share more inputs.

Fig 1. shows to compute a column of 4 results will need to have 11 loads and 1 store.

![image]()


Fig 2. show to compute a squre of 2*2 results will need to have 9 loads and 1 store.
![image]()
 

Fig 3
![image]()
Compare Fig 2 and Fig 3. we can know that calculating more results per thread can increase arithmetic intensity

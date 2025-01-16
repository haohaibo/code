#include <cuda_runtime.h>
#include <iostream>

// 简单内核
__global__ void kernel() {
    // 每个线程执行简单操作
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        printf("Kernel launched with %d threads per block\n", blockDim.x);
    }
}

int main() {
    const int warpSize = 32; // 每个 Warp 包含 32 个线程
    for (int numWarps = 1; numWarps <= 16; ++numWarps) {
        int threadsPerBlock = numWarps * warpSize; // 根据 Warp 数配置线程数

        // 启动内核
        kernel<<<1, threadsPerBlock>>>();
        cudaDeviceSynchronize();

        // 检查是否运行成功
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) {
            std::cout << "Success with " << numWarps << " warp(s), threads per block: " 
                      << threadsPerBlock << std::endl;
        } else {
            std::cout << "Failed with " << numWarps << " warp(s), threads per block: " 
                      << threadsPerBlock << ": " << cudaGetErrorString(err) << std::endl;
        }
    }
    return 0;
}
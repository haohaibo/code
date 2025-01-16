#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel() {
    extern __shared__ int sharedMem[]; // 动态分配共享内存
}

int main() {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);

    std::cout << "Shared Memory Size (Static): " << attr.sharedSizeBytes << " bytes" << std::endl;
    return 0;
}
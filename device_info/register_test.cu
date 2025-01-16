#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel1() { int x = threadIdx.x; }
__global__ void kernel2() { int x = threadIdx.x, y = x + 1; }
__global__ void kernel3() { int x = threadIdx.x, y = x + 1, z = y + 2; }

int main() {
    cudaFuncAttributes attr;

    cudaFuncGetAttributes(&attr, kernel1);
    std::cout << "Kernel1: Registers per thread: " << attr.numRegs << std::endl;

    cudaFuncGetAttributes(&attr, kernel2);
    std::cout << "Kernel2: Registers per thread: " << attr.numRegs << std::endl;

    cudaFuncGetAttributes(&attr, kernel3);
    std::cout << "Kernel3: Registers per thread: " << attr.numRegs << std::endl;

    return 0;
}
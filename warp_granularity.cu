#include <cuda_runtime.h>
#include <iostream>

__global__ void warpTest(int *warpCount) {
    // 获取当前线程的 Warp ID
    int warpId = threadIdx.x / 32;

    // 使用原子操作统计活跃的 Warp 数量
    if (threadIdx.x % 32 == 0) {
        atomicAdd(warpCount, 1);
    }
}

int main() {
    int *d_warpCount;
    int h_warpCount = 0;

    cudaMalloc(&d_warpCount, sizeof(int));
    cudaMemset(d_warpCount, 0, sizeof(int));

    // 设置线程数为 48，观察 Warp 调度
    warpTest<<<1, 48>>>(d_warpCount);

    cudaMemcpy(&h_warpCount, d_warpCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_warpCount);

    std::cout << "(48 threads test)Active Warps: " << h_warpCount << std::endl;


    cudaMalloc(&d_warpCount, sizeof(int));
    cudaMemset(d_warpCount, 0, sizeof(int));

    // 设置线程数为 65，观察 Warp 调度
    warpTest<<<1, 65>>>(d_warpCount);

    cudaMemcpy(&h_warpCount, d_warpCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_warpCount);

    std::cout << "(65 threads test)Active Warps: " << h_warpCount << std::endl;

    return 0;
}
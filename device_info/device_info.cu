#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if(err != cudaSuccess)
    {
        std::cout << "cudaGetDeviceCount() execute fail" << std::endl;
    }

    for(int dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Maximum number of threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Maximum number of blocks per mutiprocessor: " <<  deviceProp.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "Multiprocessor count: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "Registers per multiprocessor: " << deviceProp.regsPerMultiprocessor << std::endl;
        std::cout << "Total global memory: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock << std::endl;

        // Shared memory reserved by CUDA driver per block in bytes
        std::cout << "Reserved shared memory per block(Shared memory reserved by CUDA driver per block in bytes): " << deviceProp.reservedSharedMemPerBlock << std::endl;
        std::cout << "Shared memory per multiprocessor: " << deviceProp.sharedMemPerMultiprocessor << std::endl;
    }
}
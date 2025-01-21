#include <iostream>
#include <cuda_runtime.h>

#define N 1024

__global__ void deviceCopy(int* dOut, int* dIn, int Count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = tid; i < Count/4; i += blockDim.x * gridDim.x)
    {
        reinterpret_cast<int4*>(dOut)[i] = reinterpret_cast<int4*>(dIn)[i];
    }
}

int main()
{
    int* hIn = new int[N];
    int* hOut = new int[N];
    
    for(int i = 0; i < N; ++i)
    {
        hIn[i] = i;
    }

    int* dIn;
    cudaMalloc(&dIn, N * sizeof(int));
    int* dOut;
    cudaMalloc(&dOut, N * sizeof(int));

    cudaMemcpy(dIn, hIn, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 128;

    dim3 blocksPerGrid((N/4 + threads -1) / threads);
    dim3 threadsPerBlock(threads);

    deviceCopy<<<blocksPerGrid, threadsPerBlock>>>(dOut, dIn, N);
    cudaMemcpy(hOut, dOut, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; ++i)
    {
        if((hOut[i] - hIn[i]) > 0.01)
        {
            std::cout << "output mismatch!" << std::endl;
            break;
        }
    }
}

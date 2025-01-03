#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <cmath>

#define DEBUG 0

/*
 compute C = alpha*A*B + beta*C
 MxN = MxK + KxN
*/

__global__ void sgemm(int M, int K, int N, float alpha, float beta, const float* A, const float* B, float* C)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < M && y < N)
    {
      // compute
      float c = 0;
      for(int k = 0; k < K; ++k)
      {
          c += A[x*K + k] * B[N*k + y];
      }
      // write C
      C[x*N + y] = alpha*c + beta*C[x*N + y];
    }
}

int main()
{
    int M = 4096;
    int N = 4096;
    int K = 4096;
    float alpha = 1;
    float beta = 0;
  
    auto a_host = new float[M*K];
    auto b_host = new float[K*N];
    auto c_host = new float[M*N];
    auto c_verify = new float[M*N];
    std::cout << "M=" << M << ", N=" << N << ", K=" <<K << std::endl;
    srand((unsigned)time(NULL));
#if DEBUG
    std::cout << "a_host" << std::endl;
#endif
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < K; ++j)
        {
            a_host[i*M + j] = static_cast<float>(rand() % 100);
          #if DEBUG
            std::cout << a_host[i*M + j] << " ";
          #endif
        }
        #if DEBUG
        std::cout << std::endl;
        #endif
    }

#if DEBUG
    std::cout << "b_host" << std::endl;
#endif
    for(int i = 0; i < K; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            b_host[i*K + j] = static_cast<float>(rand() % 100);
          #if DEBUG
            std::cout << b_host[i*K + j] << " ";
          #endif
        }
        #if DEBUG
        std::cout << std::endl;
        #endif
    }
#if DEBUG
    std::cout << "c_verify" << std::endl;
#endif
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            int c = 0;
            for(int k = 0; k < K; ++k)
            {
                c += a_host[i*M + k] * b_host[k*K + j];
            }
            c_verify[i*M + j] = alpha*c + beta*c_verify[i*M + j];
          #if DEBUG
            std::cout << c_verify[i*M + j] << " ";
          #endif
        }
        #if DEBUG
        std::cout << std::endl;
        #endif
    }

    float* a_device;
    float* b_device;
    float* c_device;
    cudaMalloc(&a_device, sizeof(float)*M*K);
    cudaMalloc(&b_device, sizeof(float)*K*N);
    cudaMalloc(&c_device, sizeof(float)*M*N);

    cudaMemcpy(a_device, a_host, sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, sizeof(float)*K*N, cudaMemcpyHostToDevice);

    //
    dim3 blocksPerGrid((M+31)/32, (N+31)/32);
    // use 1024 threads per block
    // threadIdx.x 32, threadIdx.y 32, threadIdx.z 1
    // one thread compute one output element of C
    dim3 threadsPerBlock(32, 32, 1);
    sgemm<<<blocksPerGrid, threadsPerBlock>>>(M, K, N, alpha, beta, a_device, b_device, c_device);

    cudaMemcpy(c_host, c_device, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    // verify
    bool flag = true;
#if 1
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            if(abs(c_verify[i*M + j]-c_host[i*M + j]>0.01))
            {
                std::cout << "(" << i << "," << j << ") " <<"diff golden - device" << c_verify[i*M + j]-c_host[i*M + j] << std::endl;
                flag = false;
            }
        }
    }
#endif

    if(flag)
    {
        std::cout << "compute pass" << std::endl;
    }else
    {
        std::cout << "compute fail" << std::endl;
    }

}

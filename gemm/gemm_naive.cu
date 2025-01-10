#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

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
for(int i = 1024; i <= 2048; i += 1024)
{
    int M = i;
    int N = i;
    int K = i;
    float alpha = 1;
    float beta = 0;
  
    auto a_host = new float[M*K];
    auto b_host = new float[K*N];
    auto c_host = new float[M*N];
    auto c_ref_host = new float[M*N];
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
    std::cout << "c_ref" << std::endl;
#endif
#if DEBUG
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            int c = 0;
            for(int k = 0; k < K; ++k)
            {
                c += a_host[i*M + k] * b_host[k*K + j];
            }
            c_ref_host[i*M + j] = alpha*c + beta*c_ref_host[i*M + j];
          #if DEBUG
            std::cout << c_ref_host[i*M + j] << " ";
          #endif
        }
        #if DEBUG
        std::cout << std::endl;
        #endif
    }
#endif

    float* a_device;
    float* b_device;
    float* c_device;
    float* c_ref_device;
    cudaMalloc(&a_device, sizeof(float)*M*K);
    cudaMalloc(&b_device, sizeof(float)*K*N);
    cudaMalloc(&c_device, sizeof(float)*M*N);
    cudaMalloc(&c_ref_device, sizeof(float)*M*N);

    cudaMemcpy(a_device, a_host, sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, sizeof(float)*K*N, cudaMemcpyHostToDevice);

/*
cublasStatus_t cublasSgemm(cublasHandle_t handle,
               cublasOperation_t transa, cublasOperation_t transb,
			   int m, int n, int k,
			   const float           *alpha,
			   const float           *A, int lda,
		       const float           *B, int ldb,										       
               const float           *beta,
		       float           *C, int ldc)
*/



    // create handle
    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
	    std::cout << "CUBLAS initialization failure" << std::endl;
    }
    stat = cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
			   N, M, K,
			   &alpha,
			   b_device, N,
		           a_device, K,
			   &beta,
			   c_ref_device, N);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
	    std::cout << "CUBLAS Sgemm execution failure" << std::endl;
    }

    //
    dim3 blocksPerGrid((M+31)/32, (N+31)/32);
    // use 1024 threads per block
    // threadIdx.x 32, threadIdx.y 32, threadIdx.z 1
    // one thread compute one output element of C
    dim3 threadsPerBlock(32, 32, 1);
    // warm up
    sgemm<<<blocksPerGrid, threadsPerBlock>>>(M, K, N, alpha, beta, a_device, b_device, c_device);

    float elapsed_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int repeat_times = 10;
    cudaEventRecord(start);
    for(int i = 0; i < repeat_times; ++i)
    {
        sgemm<<<blocksPerGrid, threadsPerBlock>>>(M, K, N, alpha, beta, a_device, b_device, c_device);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    elapsed_time /= 1000.0; // seconds

    std::cout << "Average elapsed time: " << elapsed_time/repeat_times << " second(s), performance: "
	    << (1.0e-9)*2*M*K*N*repeat_times/elapsed_time << " GFLOPS. Memory bandwith: "
        << (1.0e-9)*4*(M*K + K*N + M*N)*repeat_times/elapsed_time << " GB/s" << std::endl;

    cudaMemcpy(c_host, c_device, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    float elapsed_time_cublas;
    cudaEvent_t start_cublas, end_cublas;
    cudaEventCreate(&start_cublas);
    cudaEventCreate(&end_cublas);

    cudaEventRecord(start_cublas);
    for(int i = 0; i < repeat_times; ++i)
    {
          cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
			   N, M, K,
			   &alpha,
			   b_device, N,
		           a_device, K,
			   &beta,
			   c_ref_device, N);
    }
    cudaEventRecord(end_cublas);
    cudaEventSynchronize(start_cublas);
    cudaEventSynchronize(end_cublas);
    cudaEventElapsedTime(&elapsed_time_cublas, start_cublas, end_cublas);
    elapsed_time_cublas /= 1000.0; // seconds
    std::cout << "Cublas Average elapsed time: " << elapsed_time_cublas << " second(s), performance: "
	    << (1.0e-9)*2*M*K*N*repeat_times/elapsed_time_cublas << " GFLOPS. Memory bandwidth: "
        << (1.0e-9)*4*(M*K + K*N + M*N)*repeat_times/elapsed_time_cublas << " GB/s" << std::endl;

    cudaMemcpy(c_ref_host, c_ref_device, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
 

    // verify
    bool flag = true;
#if 1
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            if(abs(c_ref_host[i*M + j]-c_host[i*M + j]>0.01))
            {
                std::cout << "(" << i << "," << j << ") " <<"diff(cublas, custom) " << c_ref_host[i*M + j]-c_host[i*M + j] << std::endl;
                flag = false;
            }
        }
    }
#endif


    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);
    cudaFree(c_ref_device);

    delete[] a_host;
    delete[] b_host;
    delete[] c_host;
    delete[] c_ref_host;

    if(flag)
    {
        std::cout << "compute pass" << std::endl;
    }else
    {
        std::cout << "compute fail" << std::endl;
    }
}

}
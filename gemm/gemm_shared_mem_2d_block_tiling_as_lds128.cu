#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>
#include <stdio.h>

#define DEBUG 0

// default set to 10
#define REPEAT_TIMES 10

// default set to 1
#define WARM_UP 1

/*
 compute C = alpha*A*B + beta*C
 MxN = MxK + KxN
*/

template<const uint BM, const uint BK, const uint BN, const uint TM, const uint TN>
__global__ void sgemm(int M, int K, int N, float alpha, float beta, const float* A, const float* B, float* C)
{
    A += blockIdx.x * BM * K;
    B += blockIdx.y * BN;
    C += blockIdx.x * BM * N + blockIdx.y * BN;

    // one thread block compute thread block size of C
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    //int xa = threadIdx.x / (BK/TN);
    //int ya = threadIdx.x % (BK/TN);

    int elementsLoadPerThread = (BK * TM * TN) / BM;

    int xb = threadIdx.x / (BN/TN);
    int yb = threadIdx.x % (BN/TN);


    int colA = threadIdx.x % BK;
    int rowA = threadIdx.x / BK;

    int colB = threadIdx.x % BN;
    int rowB = threadIdx.x / BN;

    float c[TM*TN] = {0.0};

    float regAs[TM] = {0.0};
    float regBs[TN] = {0.0};
    for(int block = 0; block < K/BK; ++block)
    {
    #if 0
        for(int t = 0; t < TM; ++t)
        {
            //As[xa * BK + ya * TN + t] = A[xa * K + yb * TN + t];

            As[(xb * TM + t) * BK + yb] = A[(xb * TM + t) * K + yb];
        }
    #endif
        for(int t = 0; t < elementsLoadPerThread; ++t)
        {
            //As[((rowA * elementsLoadPerThread + t) * BK) + colA] = A[((rowA * elementsLoadPerThread + t) * K) + colA];

            As[(colA * BM) + (rowA * elementsLoadPerThread + t)] = A[((rowA * elementsLoadPerThread + t) * K) + colA];
        }

        for(int t = 0; t < elementsLoadPerThread; ++t)
        {
            //Bs[xb * BN + yb * TN + t] = B[xb * N + yb * TN + t];

            //Bs[(t * BN) + threadIdx.x] = B[(t * N) + threadIdx.x];

            Bs[((rowB * elementsLoadPerThread + t) * BN) + colB] = B[((rowB * elementsLoadPerThread + t) * N) + colB];
        }

        // sync to make sure shared memory are fully loaded by all threads in a thread block
        __syncthreads();

        A += BK;
        B += BK * N;

        // inner block loop
        for(int k = 0; k < BK; ++k)
        {
            for(int i = 0; i < TM; ++i)
            {
                //regAs[i] = As[(xb * TM + i) * BK + k];

                regAs[i] = As[(xb * TM + i) + k * BM];
            }

            for(int j = 0; j < TN; ++j)
            {
                regBs[j] = Bs[(yb * TN + j) + k * BN];
            }

            for(int tm = 0; tm < TM; ++tm)
            {
                for(int tn = 0; tn < TN; ++tn)
                {
                    //c[tm * TN + tn] += As[(xb * TM + tm) * BK + k] * Bs[(yb * TN + tn) + k * BN];
                    c[tm * TN + tn] += regAs[tm] * regBs[tn];
                }
            }
        }
        // sync to make sure all threads in a thread block have completed the partial sum
        // shared memory are all consumed before the next load
        __syncthreads();
    }

    for(int tm = 0; tm < TM; ++tm)
    {
        for(int tn = 0; tn < TN; ++tn)
        {
            C[(xb * TM + tm) * N + yb * TN + tn] = alpha*c[tm * TN + tn] + beta*C[(xb * TM + tm) * N + yb * TN + tn];
        }
    }

}

int main()
{
for(int i = 1024; i <= 4096; i += 1024)
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
            a_host[i*M + j] = static_cast<float>(rand() % 5);
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
            b_host[i*K + j] = static_cast<float>(rand() % 5);
          #if DEBUG
            std::cout << b_host[i*K + j] << " ";
          #endif
        }
        #if DEBUG
        std::cout << std::endl;
        #endif
    }
#if DEBUG
    std::cout << "c_ref_host" << std::endl;
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
			   const float *alpha,
			   const float *A, int lda,
		       const float *B, int ldb,										       
               const float *beta,
		       float *C, int ldc)
*/



    // create handle
    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
	    std::cout << "CUBLAS initialization failure" << std::endl;
    }
#if WARM_UP
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
#endif

    // 1 thread compute 8x8 output elements acrossing M dimension
    const uint TM = 8, TN = 8;
    const uint BK = 8;

    float elapsed_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    if(M >= 128 && N >= 128)
    {
    const uint BM = 128, BN = 128;

    //
    dim3 blocksPerGrid((M+BM-1)/BM, (N+BN-1)/BN);
    // one thread compute 8x8 output element of C
    dim3 threadsPerBlock((BM*BN)/(TM*TN));

#if WARM_UP
    // warm up
    sgemm<BM, BK, BN, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(M, K, N, alpha, beta, a_device, b_device, c_device);
#endif

    cudaEventRecord(start);
    for(int i = 0; i < REPEAT_TIMES; ++i)
    {
        sgemm<BM, BK, BN, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(M, K, N, alpha, beta, a_device, b_device, c_device);
    }
    cudaEventRecord(end);

    }else
    {
    const uint BM = 64, BN = 64;
    //
    dim3 blocksPerGrid((M+BM-1)/BM, (N+BN-1)/BN);
    // one thread compute 8x8 output element of C
    dim3 threadsPerBlock((BM*BN)/(TM*TN));

#if WARM_UP
    // warm up
    sgemm<BM, BK, BN, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(M, K, N, alpha, beta, a_device, b_device, c_device);
#endif

    cudaEventRecord(start);
    for(int i = 0; i < REPEAT_TIMES; ++i)
    {
        sgemm<BM, BK, BN, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(M, K, N, alpha, beta, a_device, b_device, c_device);
    }
    cudaEventRecord(end);
    }


    cudaError_t err;
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "kernel launch or execute failure" << std::endl;
        std::cout << cudaGetErrorString(err) << std::endl;
    }

    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    elapsed_time /= 1000.0; // seconds

    std::cout << "Average elapsed time: " << elapsed_time/REPEAT_TIMES << " second(s), performance: "
	    << (1.0e-9)*2*M*K*N*REPEAT_TIMES/elapsed_time << " GFLOPS. Memory bandwith: "
        << (1.0e-9)*4*(M*K + K*N + M*N)*REPEAT_TIMES/elapsed_time << " GB/s" << std::endl;

    cudaMemcpy(c_host, c_device, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    float elapsed_time_cublas;
    cudaEvent_t start_cublas, end_cublas;
    cudaEventCreate(&start_cublas);
    cudaEventCreate(&end_cublas);

    cudaEventRecord(start_cublas);
    for(int i = 0; i < REPEAT_TIMES; ++i)
    {
    stat = cublasSgemm(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
			   N, M, K,
			   &alpha,
			   b_device, N,
		           a_device, K,
			   &beta,
			   c_ref_device, N);
    }

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
	    std::cout << "CUBLAS Sgemm execution failure" << std::endl;
    }
    cudaEventRecord(end_cublas);
    cudaEventSynchronize(start_cublas);
    cudaEventSynchronize(end_cublas);
    cudaEventElapsedTime(&elapsed_time_cublas, start_cublas, end_cublas);
    elapsed_time_cublas /= 1000.0; // seconds
    std::cout << "Cublas Average elapsed time: " << elapsed_time_cublas << " second(s), performance: "
	    << (1.0e-9)*2*M*K*N*REPEAT_TIMES/elapsed_time_cublas << " GFLOPS. Memory bandwidth: "
        << (1.0e-9)*4*(M*K + K*N + M*N)*REPEAT_TIMES/elapsed_time_cublas << " GB/s" << std::endl;

    cudaMemcpy(c_ref_host, c_ref_device, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
 

#if DEBUG
    std::cout << "c_host" << std::endl;
#endif
#if DEBUG
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            std::cout << c_host[i*M + j] << " ";
        }
        std::cout << std::endl;
    }
#endif


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

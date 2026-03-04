#include "utils/cuda_utils.cuh"
#include <cublas_v2.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

int main(){
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    cuda_utils::init_matrix(h_A, M, K);
    cuda_utils::init_matrix(h_B, K, N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // warmup
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, 
                &alpha, 
                d_B, N,  // Matrix B
                d_A, K,  // Matrix A
                &beta, 
                d_C, N); // Matrix C
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iterations = 20;
    cudaEventRecord(start);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    N, M, K, 
                    &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / num_iterations;

    double tflops_total = 2.0 * M * N * K;
    double achieved_tflops = (tflops_total / (avg_ms * 1e-3)) / 1e12;

    printf("average time: %f ms\n", avg_ms);
    printf("achieved Compute: %f TFLOPS\n", achieved_tflops);
    printf("theoretical Max Compute: ~31.2 TFLOPS\n");

    cublasDestroy(handle);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
}
/*
compile using: nvcc 12_matmul_cuBLAS.cu -arch=sm_86 -O3 -lcublas && ./a.out
average time: 0.160000 ms
achieved Compute: 13.421773 TFLOPS
theoretical Max Compute: ~31.2 TFLOPS

*/
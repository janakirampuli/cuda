#include "utils/cuda_utils.cuh"

#include <iostream>

#define BLOCK_SIZE 16

__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int j = threadIdx.x + blockDim.x * blockIdx.x; // Fast-changing index (cols)
    int i = threadIdx.y + blockDim.y * blockIdx.y; // Slow-changing index (rows)

    if (i < m && j < n){
        float sum = 0.0;
        for(int a = 0; a < k; a++){
            sum += A[i*k + a] * B[a*n + j];
        }
        C[i*n + j] = sum;
    }
}

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


    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(cuda_utils::ceil_div(N, BLOCK_SIZE), cuda_utils::ceil_div(M, BLOCK_SIZE), 1);

    // warmup
    matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iterations = 20;
    cudaEventRecord(start);
    for (int iter = 0; iter < num_iterations; iter++) {
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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


    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);

    return 0;
}

/*

output:
average time: 1.156454 ms
achieved Compute: 1.856955 TFLOPS
theoretical Max Compute: ~31.2 TFLOPS

*/
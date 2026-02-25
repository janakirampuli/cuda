#include "utils/cuda_utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

#define M 512  // Number of rows in A and C
#define K 512   // Number of columns in A and rows in B
#define N 256  // Number of columns in B and C
#define BLOCK_SIZE 32


// Example 3x2 @ 2x4 = 3x4 -> (M x K) @ (K x N) = (M x N)
// A = [[1, 2], 
//      [3, 4], 
//      [5, 6]]

// B = [[7, 8, 9, 10],
//      [11, 12, 13, 14]]

// C = A * B = [[1*7 + 2*11, 1*8 + 2*12, 1*9 + 2*13, 1*10 + 2*14],
//              [3*7 + 4*11, 3*8 + 4*12, 3*9 + 4*13, 3*10 + 4*14],
//              [5*7 + 6*11, 5*8 + 6*12, 5*9 + 6*13, 5*10 + 6*14]]

// C = [[29, 32, 35, 38],
//      [65, 72, 79, 86],
//      [101, 112, 123, 134]]


void matmul_cpu(float *A, float *B, float *C, int m, int k, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            float sum = 0.0;
            for(int a = 0; a < k; a++){
                sum += A[i*k + a] * B[a*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < m && j < n){
        float sum = 0.0;
        for(int a = 0; a < k; a++){
            sum += A[i*k + a] * B[a*n + j];
        }
        C[i*n + j] = sum;
    }
}

int main(){
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    srand(time(NULL));
    cuda_utils::init_matrix(h_A, M, K);
    cuda_utils::init_matrix(h_B, K, N);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(cuda_utils::ceil_div(M, BLOCK_SIZE), cuda_utils::ceil_div(N, BLOCK_SIZE));

    printf("performing warmup \n");
    for(int i = 0; i < 3; i++){
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }

    //benchmark cpu
    printf("benchmarking cpu \n");
    double cpu_time = 0.0;
    for (int i = 0; i < 20; i ++){
        double start_t = cuda_utils::time_sec();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end_t = cuda_utils::time_sec();
        cpu_time += end_t - start_t;
    }

    cpu_time /= 20.0;

    printf("benchmarking gpu \n");
    double gpu_time = 0.0;
    for(int i = 0; i < 20; i++){
        double start_t = cuda_utils::time_sec();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        double end_t = cuda_utils::time_sec();
        gpu_time += end_t - start_t;
    }

    gpu_time /= 20.0;

    printf("CPU avg time: %f ms\n", cpu_time*1000);
    printf("GPU avg time: %f ms\n", gpu_time*1000);
    printf("speedup: %fx\n", cpu_time/gpu_time);

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
    std::size_t bad_idx = 0;
    bool correct = cuda_utils::allclose_f32(h_C_cpu, h_C_gpu, static_cast<std::size_t>(M) * N, 1e-4f, 0.0f, &bad_idx);
    if (!correct) {
        std::cout << bad_idx << " cpu: " << h_C_cpu[bad_idx] << " != " << h_C_gpu[bad_idx] << std::endl;
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");


    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaGetLastError());

    return 0;
    
}

/*

performing warmup 
benchmarking cpu 
benchmarking gpu 
CPU avg time: 188.603707 ms
GPU avg time: 0.655357 ms
speedup: 287.787911x
Results are correct

*/
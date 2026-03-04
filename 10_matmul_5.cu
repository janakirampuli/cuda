#include "utils/cuda_utils.cuh"

#include <iostream>
#include <cstdio>
#include <cstdlib>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// The shared memory tile sizes
#define BM 64 // Block M
#define BN 64 // Block N
#define BK 16 // Block K


// The thread tile sizes (Each thread computes a 4x4 grid of C)
#define TM 4 // Thread M
#define TN 4 // Thread N

__global__ void matmul_2d_coarsened_gpu(float *A, float *B, float *C, int m, int k, int n) {

    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // base row, col in output C(64*64)
    int row_start = by * BM;
    int col_start = bx * BN;

    int tid = ty * blockDim.x + tx; 

    // REGISTER CACHE: each thread now holds a 4x4 2D grid of output values
    float sum[TM][TN] = {0.0f};

    // REGISTER CACHE: to hold the fetched values of A and B
    float reg_A[TM];
    float reg_B[TN];

    int num_tiles = (k + BK - 1) / BK;

     for (int t = 0; t < num_tiles; ++t) {
        // load A into s_A
        for (int i = 0; i < (BM * BK) / 256; ++i) {
            int linear_idx = tid + i * 256;
            int a_row = linear_idx / BK;
            int a_col = linear_idx % BK;
            int g_row = row_start + a_row;
            int g_col = t * BK + a_col;
            
            if (g_row < m && g_col < k) s_A[a_row][a_col] = A[g_row * k + g_col];
            else s_A[a_row][a_col] = 0.0f;
        }

        // load B into s_B
        for (int i = 0; i < (BK * BN) / 256; ++i) {
            int linear_idx = tid + i * 256;
            int b_row = linear_idx / BN;
            int b_col = linear_idx % BN;
            int g_row = t * BK + b_row;
            int g_col = col_start + b_col;
            
            if (g_row < k && g_col < n) s_B[b_row][b_col] = B[g_row * n + g_col];
            else s_B[b_row][b_col] = 0.0f;
        }

        __syncthreads();

        for (int k_idx = 0; k_idx < BK; ++k_idx) {
            
            // load TM (4) elements of A into registers
            for (int i = 0; i < TM; ++i) {
                // Notice the layout: threads stride by 16 (the blockDim.y)
                reg_A[i] = s_A[ty + i * 16][k_idx]; 
            }
            
            // load TN (4) elements of B into registers
            for (int j = 0; j < TN; ++j) {
                reg_B[j] = s_B[k_idx][tx + j * 16]; 
            }
            
            // multiply the two register vectors (Outer Product)
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    sum[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
     }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int g_row = row_start + ty + i * 16;
            int g_col = col_start + tx + j * 16;
            if (g_row < m && g_col < n) {
                C[g_row * n + g_col] = sum[i][j];
            }
        }
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


    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 gridDim(cuda_utils::ceil_div(N, BN), cuda_utils::ceil_div(M, BM), 1);

    // warmup
    matmul_2d_coarsened_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iterations = 20;
    cudaEventRecord(start);
    for (int iter = 0; iter < num_iterations; iter++) {
        matmul_2d_coarsened_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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

compile using: nvcc -arch=sm_86 10_matmul_5.cu --ptxas-options=-v && ./a.out

ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z23matmul_2d_coarsened_gpuPfS_S_iii' for 'sm_86'
ptxas info    : Function properties for _Z23matmul_2d_coarsened_gpuPfS_S_iii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 72 registers, 8192 bytes smem, 388 bytes cmem[0]
average time: 0.282470 ms
achieved Compute: 7.602508 TFLOPS
theoretical Max Compute: ~31.2 TFLOPS
*/
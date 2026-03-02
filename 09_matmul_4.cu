#include "utils/cuda_utils.cuh"

#include <iostream>

// 16*16 = 256 threads per block, each thread computes 4 elements(COARSE_FACTOR)
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

#define TILE_M 16
#define TILE_N 64
#define TILE_K 16

#define COARSE_FACTOR 4

__global__ void matmul_1d_coarsened_gpu(float *A, float *B, float *C, int m, int k, int n) {

    __shared__ float s_A[TILE_M][TILE_K];
    __shared__ float s_B[TILE_K][TILE_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // row remains same as tx++
    int row = ty + TILE_M * by;

    int tid = blockDim.x * ty + tx;

    // each thread holds 4 output values
    float sum[COARSE_FACTOR] = {0.0f};

    int num_tiles = (k + TILE_K - 1) / TILE_K;

    for(int t = 0; t < num_tiles; t++){
        // load A into s_A (256 threads load 256 elements -> 1 load per thread)
        if(row < m && (t * TILE_K + tx) < k){
            s_A[ty][tx] = A[tx + row * k + t * TILE_K];
        }
        else{
            s_A[ty][tx] = 0.0f;
        }
        // load B into s_B (256 threads load 1024 elements -> 4 loads per thread)
        for(int i = 0; i < COARSE_FACTOR; i++){
            int linear_idx = tid + i * 256;
            int b_row = linear_idx / TILE_N;
            int b_col = linear_idx % TILE_N;

            int global_b_row = b_row + t * TILE_K;
            int global_b_col = b_col + bx * TILE_N;

            if (global_b_row < k && global_b_col < n) {
                s_B[b_row][b_col] = B[global_b_row * n + global_b_col];
            } else {
                s_B[b_row][b_col] = 0.0f;
            }
        }

        __syncthreads();

        for(int k_idx = 0; k_idx < TILE_K; k_idx++){
            // load once from shared mem
            float a = s_A[ty][k_idx];
            // compute 4 values using a
            for(int c = 0; c < COARSE_FACTOR; c++){
                sum[c] += a * s_B[k_idx][tx * COARSE_FACTOR + c];
            }
        }
        __syncthreads();
    }

    for(int c = 0; c < COARSE_FACTOR; c++){
        int col = bx * TILE_N + tx * COARSE_FACTOR + c;
        if(row < m && col < n){
            C[row * n + col] = sum[c];
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
    dim3 gridDim(cuda_utils::ceil_div(N, TILE_N), cuda_utils::ceil_div(M, TILE_M), 1);

    // warmup
    matmul_1d_coarsened_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iterations = 20;
    cudaEventRecord(start);
    for (int iter = 0; iter < num_iterations; iter++) {
        matmul_1d_coarsened_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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

compile using: nvcc -arch=sm_86 09_matmul_4.cu --ptxas-options=-v && ./a.out

ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z23matmul_1d_coarsened_gpuPfS_S_iii' for 'sm_86'
ptxas info    : Function properties for _Z23matmul_1d_coarsened_gpuPfS_S_iii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 48 registers, 5120 bytes smem, 388 bytes cmem[0]
average time: 0.443648 ms
achieved Compute: 4.840512 TFLOPS
theoretical Max Compute: ~31.2 TFLOPS
*/
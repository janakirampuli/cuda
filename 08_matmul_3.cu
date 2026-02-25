#include "utils/cuda_utils.cuh"

#include <iostream>

#define BLOCK_SIZE 16

__global__ void matmul_shared_gpu(float *A, float *B, float *C, int m, int k, int n) {

    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // row remains same as tx++
    int row = ty + BLOCK_SIZE * by;
    int col = tx + BLOCK_SIZE * bx;

    float sum = 0.0f;

    int num_tiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int t = 0; t < num_tiles; t++){
        if(row < m && (t * BLOCK_SIZE + tx) < k){
            // using col as tx as we want row to be same
            s_A[ty][tx] = A[tx + row * k + t * BLOCK_SIZE];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        if(col < n && (t * BLOCK_SIZE + ty) < n){
            s_B[ty][tx] = B[(t * BLOCK_SIZE + ty) * n + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }
        // blocking the threads in that block till cache is fully populated
        __syncthreads();

        for(int i = 0; i < BLOCK_SIZE; i++){
            sum += s_A[ty][i] * s_B[i][tx];
        }
        // sync threads before loading next tile
        __syncthreads();
    }
    if (row < m && col < n){
        C[row * n + col] = sum;
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
    matmul_shared_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iterations = 20;
    cudaEventRecord(start);
    for (int iter = 0; iter < num_iterations; iter++) {
        matmul_shared_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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

compile using: nvcc -arch=sm_86 08_matmul_3.cu --ptxas-options=-v && ./a.out

for BLOCK_SIZE = 16:

output:
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z17matmul_shared_gpuPfS_S_iii' for 'sm_86'
ptxas info    : Function properties for _Z17matmul_shared_gpuPfS_S_iii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 36 registers, 2048 bytes smem, 388 bytes cmem[0]
average time: 0.872192 ms
achieved Compute: 2.462168 TFLOPS
theoretical Max Compute: ~31.2 TFLOPS


BLOCK_SIZE = 16 => 
2*16*16*4Bytes = 2KB
shared memory per block => 48KB
shared memory per SM => 100KB

1. shared mem used by this kernel: 2048B/block(s_A, s_B) + 1024/block(CUDA runtime) = 3072B/block
now we can have max 102400/3072 = 33 blocks. so we can have max 33 bocks per SM

2. but we have 1536 max threads per SM, our block has 256 threads => max 6 blocks

3. we used 30 registers => 30 regs per thread * 36 threads per warp => 1080 regs per warp
we have 256 threads / 32 = 8 warps per block
1080 regs per warp * 8 warps per block => 8640 regs per block
we have max 65536 regs per SM => max 7 blocks

so we have can have max 6 blocks per SM

max warps we can have is 48 per SM
we have 6 blocks
each block has 256/32 = 8 warps
so total 48 warps
we're at full ocuupancy

for BLOCK_SIZE = 32:
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z17matmul_shared_gpuPfS_S_iii' for 'sm_86'
ptxas info    : Function properties for _Z17matmul_shared_gpuPfS_S_iii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 38 registers, 8192 bytes smem, 388 bytes cmem[0]
average time: 0.937048 ms
achieved Compute: 2.291754 TFLOPS
theoretical Max Compute: ~31.2 TFLOPS
*/
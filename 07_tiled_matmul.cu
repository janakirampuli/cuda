#include <cuda_runtime_api.h>
#include <iostream>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

__global__ void tiled_matmul_gpu(float *A, float *B, float *C, int M, int K, int N) {

    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = ty + by * TILE_SIZE;
    int j = tx + bx * TILE_SIZE;

    float sum = 0.0;

    for(int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++){
        if(i < M && tile * TILE_SIZE + tx < K){
            sharedA[ty][tx] = A[i*K + tile*TILE_SIZE + tx];
        }
        else sharedA[ty][tx] = 0.0f;

        if(j < N && tile * TILE_SIZE + ty < K){
            sharedB[ty][tx] = B[(tile*TILE_SIZE + ty)*N + j];
        }
        else sharedB[ty][tx] = 0.0f;

        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++){
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        __syncthreads();
    }
    if (i < M && j < N){
        C[i*N + j] = sum;
    }
}

int main(){

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    tiled_matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;
}

/*

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                             Name                           
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------
    100.0           870242          1  870242.0  870242.0    870242    870242          0.0  tiled_matmul_gpu(float *, float *, float *, int, int, int)

*/
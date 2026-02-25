#include "utils/cuda_utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define N 10000000
#define BLOCK_SIZE 256 // no of threads per block

void vector_add_cpu(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i ++){
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}

// h_ -> host, d_ -> device

int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    srand(time(NULL));
    cuda_utils::init_vector(h_a, N);
    cuda_utils::init_vector(h_b, N);

    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // ceil_div of vector length, num_of_threads
    int num_blocks = cuda_utils::ceil_div(N, BLOCK_SIZE);

    // warmup
    printf("performing warmup \n");
    for (int i = 0; i < 3; i++){
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    //benchmark cpu
    printf("benchmarking cpu \n");
    double cpu_time = 0.0;
    for (int i = 0; i < 20; i ++){
        double start_t = cuda_utils::time_sec();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_t = cuda_utils::time_sec();
        cpu_time += end_t - start_t;
    }

    cpu_time /= 20.0;

    printf("benchmarking gpu \n");
    double gpu_time = 0.0;
    for(int i = 0; i < 20; i++){
        // cudaMemset(d_c, 0, size);
        double start_t = cuda_utils::time_sec();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        double end_t = cuda_utils::time_sec();
        gpu_time += end_t - start_t;
    }

    gpu_time /= 20.0;

    printf("CPU avg time: %f ms\n", cpu_time*1000);
    printf("GPU avg time: %f ms\n", gpu_time*1000);
    printf("speedup: %fx\n", cpu_time/gpu_time);

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));
    bool correct = cuda_utils::allclose_f32(h_c_cpu, h_c_gpu, N, /*atol=*/1e-5f);
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;

}

/*

performing warmup 
benchmarking cpu 
benchmarking gpu 
CPU avg time: 31.187655 ms
GPU avg time: 0.250523 ms
speedup: 124.490213x
Results are correct

*/
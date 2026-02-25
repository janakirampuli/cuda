#include "utils/cuda_utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

#define N 10000000
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

void vector_add_cpu(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i ++){
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < nx && j < ny && k < nz){
        const int idx = i + j * nx + k * nx * ny;
        c[idx] = a[idx] + b[idx];
    }
}


// h_ -> host, d_ -> device

int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;
    float *d_a, *d_b, *d_c_1d, *d_c_3d;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu_1d = (float*)malloc(size);
    h_c_gpu_3d = (float*)malloc(size);


    srand(time(NULL));
    cuda_utils::init_vector(h_a, N);
    cuda_utils::init_vector(h_b, N);

    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c_1d, size));
    CUDA_CHECK(cudaMalloc(&d_c_3d, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // ceil_div of vector length, num_of_threads
    int num_blocks_1d = cuda_utils::ceil_div(N, BLOCK_SIZE_1D);

    int nx = 1000, ny = 100, nz = 100;

    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);

    dim3 num_blocks_3d(
        cuda_utils::ceil_div(nx, static_cast<int>(block_size_3d.x)),
        cuda_utils::ceil_div(ny, static_cast<int>(block_size_3d.y)),
        cuda_utils::ceil_div(nz, static_cast<int>(block_size_3d.z))
    );

    // warmup
    printf("performing warmup \n");
    for (int i = 0; i < 3; i++){
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
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

    printf("benchmarking gpu 1d \n");
    double gpu_1d_time = 0.0;
    for(int i = 0; i < 20; i++){
        double start_t = cuda_utils::time_sec();
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        double end_t = cuda_utils::time_sec();
        gpu_1d_time += end_t - start_t;
    }

    gpu_1d_time /= 20.0;

    CUDA_CHECK(cudaMemcpy(h_c_gpu_1d, d_c_1d, size, cudaMemcpyDeviceToHost));
    std::size_t bad_idx_1d = 0;
    bool correct_1d = cuda_utils::allclose_f32(h_c_cpu, h_c_gpu_1d, N, /*atol=*/1e-5f, /*rtol=*/0.0f, &bad_idx_1d);
    if (!correct_1d) {
        std::cout << bad_idx_1d << " cpu: " << h_c_cpu[bad_idx_1d] << " != " << h_c_gpu_1d[bad_idx_1d] << std::endl;
    }
    printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");

    printf("benchmarking gpu 3d \n");
    double gpu_3d_time = 0.0;
    for(int i = 0; i < 20; i++){
        double start_t = cuda_utils::time_sec();
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        CUDA_CHECK(cudaDeviceSynchronize());
        double end_t = cuda_utils::time_sec();
        gpu_3d_time += end_t - start_t;
    }

    gpu_3d_time /= 20.0;

    CUDA_CHECK(cudaMemcpy(h_c_gpu_3d, d_c_3d, size, cudaMemcpyDeviceToHost));
    std::size_t bad_idx_3d = 0;
    bool correct_3d = cuda_utils::allclose_f32(h_c_cpu, h_c_gpu_3d, N, /*atol=*/1e-5f, /*rtol=*/0.0f, &bad_idx_3d);
    if (!correct_3d) {
        std::cout << bad_idx_3d << " cpu: " << h_c_cpu[bad_idx_3d] << " != " << h_c_gpu_3d[bad_idx_3d] << std::endl;
    }
    printf("3D Results are %s\n", correct_3d ? "correct" : "incorrect");

    printf("CPU avg time: %f ms\n", cpu_time*1000);
    printf("GPU 1d avg time: %f ms\n", gpu_1d_time*1000);
    printf("GPU 3d avg time: %f ms\n", gpu_3d_time*1000);

    printf("speedup (cpu vs gpu 1d): %fx\n", cpu_time/gpu_1d_time);
    printf("speedup (cpu vs gpu 3d): %fx\n", cpu_time/gpu_3d_time);
    printf("speedup (gpu 1d vs gpu 3d): %fx\n", gpu_1d_time/gpu_3d_time);

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c_1d));
    CUDA_CHECK(cudaFree(d_c_3d));

    return 0;

}

/*

performing warmup 
benchmarking cpu 
benchmarking gpu 1d 
1D Results are correct
benchmarking gpu 3d 
3D Results are correct
CPU avg time: 31.278278 ms
GPU 1d avg time: 0.238483 ms
GPU 3d avg time: 0.264208 ms
speedup (cpu vs gpu 1d): 131.155388x
speedup (cpu vs gpu 3d): 118.385008x
speedup (gpu 1d vs gpu 3d): 0.902632x

1d is faster(no need to compute/maintain all variables in registry)
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
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
    int k = blockIdx.z * blockDim.z + threadIdx.y;
    if (i < nx && j < ny && k < nz){
        int idx = i + j * nx + k * ny * nz;
        if(idx < nx*ny*nz){
            c[idx] = a[idx] + b[idx];
        }
    }
}


void init_vector(float *vec, int n){
    for(int i = 0; i < n; i++){
        vec[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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
    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_1d, size);
    cudaMalloc(&d_c_3d, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // ceil_div of vector length, num_of_threads
    int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    int nx = 100, ny = 100, nz = 1000;

    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);

    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z
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
        double start_t = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_t = get_time();
        cpu_time += end_t - start_t;
    }

    cpu_time /= 20.0;

    printf("benchmarking gpu 1d \n");
    double gpu_1d_time = 0.0;
    for(int i = 0; i < 20; i++){
        double start_t = get_time();
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        cudaDeviceSynchronize();
        double end_t = get_time();
        gpu_1d_time += end_t - start_t;
    }

    gpu_1d_time /= 20.0;

    cudaMemcpy(h_c_gpu_1d, d_c_1d, size, cudaMemcpyDeviceToHost);
    bool correct_1d = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu_1d[i]) > 1e-4) {
            correct_1d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_1d[i] << std::endl;
            break;
        }
    }
    printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");

    printf("benchmarking gpu 3d \n");
    double gpu_3d_time = 0.0;
    for(int i = 0; i < 20; i++){
        double start_t = get_time();
        vector_add_gpu_3d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_t = get_time();
        gpu_3d_time += end_t - start_t;
    }

    gpu_3d_time /= 20.0;

    cudaMemcpy(h_c_gpu_3d, d_c_3d, size, cudaMemcpyDeviceToHost);
    bool correct_3d = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-4) {
            correct_3d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_3d[i] << std::endl;
            break;
        }
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
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1d);
    cudaFree(d_c_3d);

    return 0;

}

/*

performing warmup 
benchmarking cpu 
benchmarking gpu 1d 
1D Results are correct
benchmarking gpu 3d 
100 cpu: 1.72846 != 0
3D Results are incorrect
CPU avg time: 31.103964 ms
GPU 1d avg time: 0.238139 ms
GPU 3d avg time: 0.046178 ms
speedup (cpu vs gpu 1d): 130.612532x
speedup (cpu vs gpu 3d): 673.568181x
speedup (gpu 1d vs gpu 3d): 5.156995x

*/
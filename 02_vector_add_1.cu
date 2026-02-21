// https://github.com/Infatoshi/cuda-course/blob/master/05_Writing_your_First_Kernels/02%20Kernels/00_vector_add_v1.cu

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime_api.h>

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
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // ceil_div of vector length, num_of_threads
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

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
        double start_t = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_t = get_time();
        cpu_time += end_t - start_t;
    }

    cpu_time /= 20.0;

    printf("benchmarking gpu \n");
    double gpu_time = 0.0;
    for(int i = 0; i < 20; i++){
        // cudaMemset(d_c, 0, size);
        double start_t = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_t = get_time();
        gpu_time += end_t - start_t;
    }

    gpu_time /= 20.0;

    printf("CPU avg time: %f ms\n", cpu_time*1000);
    printf("GPU avg time: %f ms\n", gpu_time*1000);
    printf("speedup: %fx\n", cpu_time/gpu_time);

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}

/*

performing warmup 
benchmarking cpu 
benchmarking gpu 
CPU avg time: 31.158035 ms
GPU avg time: 0.250917 ms
speedup: 124.176487x
Results are correct

*/
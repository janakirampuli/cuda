#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime_api.h>
#include<nvtx3/nvToolsExt.h>
#include<iostream>

#define N 1024
#define BLOCK_SIZE 32

__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < m && j < n){
        float sum = 0.0;
        for(int a = 0; a < k; a++){
            sum += A[i*k + a] * B[a*n + j];
        }
        C[i*n + j] = sum;
    }
}

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main(){

    float *A, *B, *C;
    size_t size = N * N * sizeof(float);

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    srand(time(NULL));
    init_matrix(A, N, N);
    init_matrix(B, N, N);

    nvtxRangePush("matmul");
    float *d_A, *d_B, *d_C;

    nvtxRangePush("memory allocation");
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    nvtxRangePop();

    nvtxRangePush("memory copy H2D");
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    nvtxRangePop();

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    nvtxRangePush("kernel execution");
    matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, N, N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePush("memory copy D2H");
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePush("memory deallocation");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();

    nvtxRangePop();

    free(A);
    free(B);
    free(C);
    return 0;
}

/*

Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style          Range        
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  --------------------
     50.0        358526875          1  358526875.0  358526875.0  358526875  358526875          0.0  PushPop  :matmul             
     49.4        353865104          1  353865104.0  353865104.0  353865104  353865104          0.0  PushPop  :memory allocation  
      0.3          2212473          1    2212473.0    2212473.0    2212473    2212473          0.0  PushPop  :memory copy D2H    
      0.2          1231597          1    1231597.0    1231597.0    1231597    1231597          0.0  PushPop  :kernel execution   
      0.1           878932          1     878932.0     878932.0     878932     878932          0.0  PushPop  :memory copy H2D    
      0.0           335229          1     335229.0     335229.0     335229     335229          0.0  PushPop  :memory deallocation

[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)           Name         
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ----------------------
     83.2        398468212         21  18974676.8  10055947.0      1028  240965055   51349178.5  poll                  
     16.2         77592071        486    159654.5      8178.5      1015    4376809     400724.5  ioctl                 
      0.3          1304352         25     52174.1      8662.0      3986     864330     170832.5  mmap64                
      0.1           662390         10     66239.0     16580.5     14235     289986     105335.9  sem_timedwait         
      0.0           184942         43      4301.0      4044.0      1665       9340       1552.6  open64                
      0.0           147630         13     11356.2      4848.0      1769      55470      15142.1  mmap                  
      0.0           139155         38      3662.0      2430.0      1066      14861       3243.3  fopen                 
      0.0           132656          3     44218.7     44755.0     41525      46376       2469.6  pthread_create        
      0.0            70144          1     70144.0     70144.0     70144      70144          0.0  pthread_cond_wait     
      0.0            55769         10      5576.9      4885.0      3546       8955       1939.8  write                 
      0.0            55540          1     55540.0     55540.0     55540      55540          0.0  fgets                 
      0.0            31505         20      1575.3      1299.0      1014       3453        645.2  fclose                
      0.0            26347          5      5269.4      3345.0      1508      12426       4374.5  fread                 
      0.0            20670          6      3445.0      3566.0      1838       4863       1316.0  open                  
      0.0            14463          3      4821.0      4673.0      3585       6205       1316.3  munmap                
      0.0            13273          6      2212.2      1886.5      1408       3873        953.2  close                 
      0.0            13164          4      3291.0      3389.0      1715       4671       1525.8  pipe2                 
      0.0            12404          2      6202.0      6202.0      5277       7127       1308.1  socket                
      0.0             8028          1      8028.0      8028.0      8028       8028          0.0  connect               
      0.0             6417          4      1604.3      1514.5      1114       2274        551.7  read                  
      0.0             4461          1      4461.0      4461.0      4461       4461          0.0  pthread_cond_broadcast
      0.0             2383          1      2383.0      2383.0      2383       2383          0.0  fcntl                 
      0.0             1975          1      1975.0      1975.0      1975       1975          0.0  bind                  
      0.0             1414          1      1414.0      1414.0      1414       1414          0.0  fwrite                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)          Name         
 --------  ---------------  ---------  ----------  ---------  --------  --------  -----------  ---------------------
     94.6         81606555          3  27202185.0    46940.0     44343  81515272   47036513.1  cudaMalloc           
      3.6          3082915          3   1027638.3   456232.0    415629   2211054    1025069.1  cudaMemcpy           
      1.4          1197665          1   1197665.0  1197665.0   1197665   1197665          0.0  cudaDeviceSynchronize
      0.4           332716          3    110905.3   124652.0     78125    129939      28511.4  cudaFree             
      0.0            29946          1     29946.0    29946.0     29946     29946          0.0  cudaLaunchKernel     

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                          Name                        
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------
    100.0          1187266          1  1187266.0  1187266.0   1187266   1187266          0.0  matmul_gpu(float *, float *, float *, int, int, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation          
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
     63.1          1166531      1  1166531.0  1166531.0   1166531   1166531          0.0  [CUDA memcpy Device-to-Host]
     36.9           682370      2   341185.0   341185.0    340609    341761        814.6  [CUDA memcpy Host-to-Device]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
      8.389      2     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Host-to-Device]
      4.194      1     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Device-to-Host]


*/
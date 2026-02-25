#include "utils/cuda_utils.cuh"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <nvtx3/nvToolsExt.h>

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

int main(){

    float *A, *B, *C;
    size_t size = N * N * sizeof(float);

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    srand(time(NULL));
    cuda_utils::init_matrix(A, N, N);
    cuda_utils::init_matrix(B, N, N);

    nvtxRangePush("matmul");
    float *d_A, *d_B, *d_C;

    nvtxRangePush("memory allocation");
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    nvtxRangePop();

    nvtxRangePush("memory copy H2D");
    CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    nvtxRangePop();

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(cuda_utils::ceil_div(N, BLOCK_SIZE), cuda_utils::ceil_div(N, BLOCK_SIZE));

    nvtxRangePush("kernel execution");
    matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, N, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    nvtxRangePush("memory copy D2H");
    CUDA_CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    nvtxRangePush("memory deallocation");
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    nvtxRangePop();

    nvtxRangePop();

    free(A);
    free(B);
    free(C);
    return 0;
}

/*

tldr: 94.2% of total time in cudaMalloc, memory allocation overhead. actual kernel is so fastt

[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style          Range       
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  -------------------
     50.0        322669887          1  322669887.0  322669887.0  322669887  322669887          0.0  PushPop  matmul             
     49.3        317859629          1  317859629.0  317859629.0  317859629  317859629          0.0  PushPop  memory allocation  
      0.3          2192516          1    2192516.0    2192516.0    2192516    2192516          0.0  PushPop  memory copy D2H    
      0.2          1388608          1    1388608.0    1388608.0    1388608    1388608          0.0  PushPop  kernel execution   
      0.1           872717          1     872717.0     872717.0     872717     872717          0.0  PushPop  memory copy H2D    
      0.1           352364          1     352364.0     352364.0     352364     352364          0.0  PushPop  memory deallocation

[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)           Name         
 --------  ---------------  ---------  ----------  ---------  --------  ---------  -----------  ----------------------
     78.5        298603661         12  24883638.4  1493920.5      1411  230612665   65463236.3  poll                  
     20.8         79153585        499    158624.4     7096.0       390    4939147     412076.4  ioctl                 
      0.4          1340837         25     53633.5     6575.0      5137     917118     181444.6  mmap64                
      0.1           533034         10     53303.4    22249.5     18068     327715      96527.2  sem_timedwait         
      0.0           177768         43      4134.1     3781.0      1302       9758       1588.3  open64                
      0.0           140300         39      3597.4     2208.0       806      14297       3452.1  fopen                 
      0.0           131474          3     43824.7    40602.0     36735      54137       9137.6  pthread_create        
      0.0           123783         13      9521.8     5524.0      1636      55130      14240.0  mmap                  
      0.0            85314          1     85314.0    85314.0     85314      85314          0.0  pthread_cond_wait     
      0.0            53739         12      4478.3     5011.5       502       7455       2152.2  write                 
      0.0            38242         33      1158.8      974.0       519       3273        598.3  fclose                
      0.0            34797         27      1288.8       47.0        46      33505       6438.5  fgets                 
      0.0            27011          7      3858.7     2199.0       123      13604       4711.6  fread                 
      0.0            19965          6      3327.5     3387.0      1130       5473       1782.9  open                  
      0.0            13832          4      3458.0     3149.0      1769       5765       1792.8  pipe2                 
      0.0            12004          3      4001.3     4082.0      3701       4221        269.2  munmap                
      0.0            11579         15       771.9      484.0       331       2234        615.1  read                  
      0.0            10971         20       548.6      536.5       186       1909        379.0  fcntl                 
      0.0            10567          2      5283.5     5283.5      4488       6079       1125.0  socket                
      0.0             9612          3      3204.0     1434.0      1365       6813       3125.7  pthread_cond_broadcast
      0.0             6743          1      6743.0     6743.0      6743       6743          0.0  connect               
      0.0             2429          2      1214.5     1214.5       969       1460        347.2  fwrite                
      0.0             2146          7       306.6      306.0       215        405         77.8  dup                   
      0.0             1655          1      1655.0     1655.0      1655       1655          0.0  bind                  
      0.0              824          1       824.0      824.0       824        824          0.0  listen                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Name         
 --------  ---------------  ---------  ----------  ---------  --------  --------  -----------  ----------------------
     94.2         78033848          3  26011282.7    65761.0     42664  77925423   44958965.8  cudaMalloc            
      3.7          3055281          3   1018427.0   435779.0    428504   2190998    1015482.8  cudaMemcpy            
      1.4          1194036          1   1194036.0  1194036.0   1194036   1194036          0.0  cudaDeviceSynchronize 
      0.4           349887          3    116629.0   126578.0     86132    137177      26937.6  cudaFree              
      0.2           183232          1    183232.0   183232.0    183232    183232          0.0  cudaLaunchKernel      
      0.0             1175          1      1175.0     1175.0      1175      1175          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                          Name                        
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------
    100.0          1192707          1  1192707.0  1192707.0   1192707   1192707          0.0  matmul_gpu(float *, float *, float *, int, int, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation          
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
     62.6          1158563      1  1158563.0  1158563.0   1158563   1158563          0.0  [CUDA memcpy Device-to-Host]
     37.4           690817      2   345408.5   345408.5    338721    352096       9457.6  [CUDA memcpy Host-to-Device]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
      8.389      2     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Host-to-Device]
      4.194      1     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Device-to-Host]

*/
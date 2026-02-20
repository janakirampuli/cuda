# basics

1. GPUs are specialized for highly parallel computations and devote more transistors to data processing units, while CPUs dedicate more transistors to data caching and flow control
2.  The CPU and the memory directly connected to it are called the host and host memory, respectively
3. A GPU and the memory directly connected to it are referred to as the device and device memory, respectively.
4. The code an application executes on the GPU is referred to as device code
5. and a function that is invoked for execution on the GPU is called a kernel
7. the GPU can be considered to be a collection of Streaming Multiprocessors (SMs) which are organized into groups called Graphics Processing Clusters (GPCs)

---

## Kernel:

A function that is invoked for execution on the GPU is called a kernel

The act of starting a kernel running is called launching the kernel

The code for a kernel is specified using the __global__ declaration specifier.

Kernels are functions with a void return type.

```cpp
// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{

}
```

---

## Thread:

smallest unit of execution in CUDA. Each thread executes the kernel code independently. When an application launches a kernel, it does so with many threads

## Thread Block:

group of threads is called a thread block

## Grid:

Thread blocks are organized into a grid. All the thread blocks in a grid have the same size and dimensions

Thread blocks and grids may be 1, 2, or 3 dimensional. These dimensions can simplify mapping of individual threads to units of work or data items.

## Thread Block Clusters:

Clusters are a group of thread blocks which, like thread blocks and grids, can be laid out in 1, 2, or 3 dimensions

## Warps:

Within a thread block, threads are organized into groups of 32 threads called warps. 

A warp executes the kernel code in a Single-Instruction Multiple-Threads (SIMT) paradigm

---

## GPU Memory:

- GPUs and CPUs both have directly attached DRAM chips
- In systems with more than one GPU, each GPU has its own memory. 
- From the perspective of device code, the DRAM attached to the GPU is called global memory, because it is accessible to all SMs in the GPU
- slowest but largest
- The DRAM attached to the CPU(s) is called system memory or host memory.
- There are CUDA APIs to allocate GPU memory, CPU memory, and to copy between allocations on the CPU and GPU, within a GPU, or between GPUs in multi-GPU systems.

## On-Chip memory:

- Each SM has its own register file and shared memory.
- These memories are part of the SM and can be accessed extremely quickly from threads executing within the SM, but they are not accessible to threads running in other SMs.

- The register file stores thread local variables which are usually allocated by the compiler
- The shared memory is accessible by all threads within a thread block or cluster.

## Cache:

- In addition to programmable memories, GPUs have both L1 and L2 caches.
- Each SM has an L1 cache which is part of the unified data cache.
- A larger L2 cache is shared by all SMs within a GPU. 

## Unified Memory:

A CUDA feature called unified memory allows applications to make memory allocations which can be accessed from CPU or GPU. 

---

## Thread and grid index:

1. threadIdx gives the index of a thread within its thread block. Each thread in a thread block will have a different index.
    Example: If you have a 1D block of 256 threads, threadIdx.x ranges from 0 to 255.
2. blockDim gives the dimensions of the thread block, which was specified in the execution configuration of the kernel launch.
    Example: If your block is 256 threads in the x-direction, blockDim.x is 256.
3. blockIdx gives the index of a thread block within the grid. Each thread block will have a different index.
    Example: If you have a 1D grid of 10 blocks, blockIdx.x ranges from 0 to 9.
4. gridDim gives the dimensions of the grid, which was specified in the execution configuration when the kernel was launched.
    Example: If your grid has 10 blocks in the x-direction, gridDim.x is 10.

Each of these intrinsics is a 3-component vector with a .x, .y, and .z member. 

Dimensions not specified by a launch configuration will default to 1.

threadIdx.x will take on values from 0 up to and including blockDim.x-1. .y and .z operate the same in their respective dimensions.

blockIdx.x will have values from 0 up to and including gridDim.x-1, and the same for .y and .z dimensions, respectively.

## dim3:

1. A simple way to specify 3D dimensions
2. Used for grid and block sizes

Example:
```cpp
    dim3 blockSize(16, 16, 1);  // 16x16x1 threads per block
    dim3 gridSize(8, 8, 1);     // 8x8x1 blocks in grid
```

## <<<>>>:

1. Special brackets for launching kernels
2. Specifies grid and block dimensions

Example:
```cpp
    addNumbers<<<gridSize, blockSize>>>(a, b, result);
```

## example:

```cpp
__global__ void vecAdd(float* A, float* B, float* C)
{
   // calculate which element this thread is responsible for computing
   int workIndex = threadIdx.x + blockDim.x * blockIdx.x

   // Perform computation
   C[workIndex] = A[workIndex] + B[workIndex];
}

int main()
{
    ...
    // A, B, and C are vectors of 1024 elements
    vecAdd<<<4, 256>>>(A, B, C);
    ...
}
```

## example:

```cpp
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x

     if(workIndex < vectorLength)
     {
         // Perform computation
         C[workIndex] = A[workIndex] + B[workIndex];
     }
}

int main(){
    ...
    // vectorLength is an integer storing number of elements in the vector
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    ...
}
```

---

## Memory management:

### unified memory

1. Memory is allocated using the cudaMallocManaged API or by declaring a variable with the __managed__ specifier.
2.  The NVIDIA Driver will make sure that the memory is accessible to the GPU or CPU whenever either tries to access it.
3. These buffers are released using cudaFree

ex.
```cpp
cudaMallocManaged(&A, vectorLength*sizeof(float));
cudaFree(A);
```

### explicit memory

1. explicitly allocates memory on the GPU using cudaMalloc
2. Memory on the GPU is freed using the same cudaFree

The CUDA API cudaMemcpy is used to copy data from a buffer residing on the CPU to a buffer residing on the GPU

ex.
```cpp
    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
        // Copy data to the GPU

        cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);

    cudaFree(devA);

    cudaFreeHost(A);
```

---

## Synchronizing CPU and GPU:

The simplest way to synchronize the GPU and a host thread is with the use of cudaDeviceSynchronize, which blocks the host thread until all previously issued work on the GPU has completed.

```cpp
cudaDeviceSynchronize()
```

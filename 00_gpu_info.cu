#include "utils/cuda_utils.cuh"

#include <iomanip>
#include <iostream>

using namespace std;

int getAttr(cudaDeviceAttr attr, int dev)
{
    int value;
    CUDA_CHECK(cudaDeviceGetAttribute(&value, attr, dev));
    return value;
}

void printBasic(cudaDeviceProp &p)
{
    cout << "\n================ BASIC DEVICE INFO ================\n";

    cout << "Name                       : " << p.name << "\n";
    cout << "Compute Capability         : " << p.major << "." << p.minor << "\n";

    cout << "\n[Execution]\n";
    cout << "Multiprocessors (SMs)      : " << p.multiProcessorCount << "\n";
    cout << "Warp Size                  : " << p.warpSize << "\n";
    cout << "Max threads per block      : " << p.maxThreadsPerBlock << "\n";
    cout << "Max threads dim            : "
              << p.maxThreadsDim[0] << " x "
              << p.maxThreadsDim[1] << " x "
              << p.maxThreadsDim[2] << "\n";
    cout << "Max grid size              : "
              << p.maxGridSize[0] << " x "
              << p.maxGridSize[1] << " x "
              << p.maxGridSize[2] << "\n";

    cout << "\n[Memory]\n";
    cout << "Total global memory (GB)   : "
              << fixed << setprecision(2)
              << p.totalGlobalMem / (1024.0*1024*1024) << "\n";

    cout << "Shared memory per block    : "
              << p.sharedMemPerBlock / 1024.0 << " KB\n";

    cout << "Shared memory per SM       : "
              << p.sharedMemPerMultiprocessor / 1024.0 << " KB\n";

    cout << "Registers per block        : " << p.regsPerBlock << "\n";

    cout << "L2 Cache Size              : "
              << p.l2CacheSize / (1024.0*1024) << " MB\n";

    cout << "Memory clock rate          : "
              << p.memoryClockRate / 1000.0 << " MHz\n";

    cout << "Memory bus width           : "
              << p.memoryBusWidth << " bits\n";

    double bandwidth =
        2.0 * p.memoryClockRate * (p.memoryBusWidth/8) / 1e6;

    cout << "Theoretical Bandwidth      : "
              << bandwidth << " GB/s\n";

    cout << "\n[Features]\n";
    cout << "Concurrent kernels         : " << p.concurrentKernels << "\n";
    cout << "Unified addressing         : " << p.unifiedAddressing << "\n";
    cout << "Managed memory             : " << p.managedMemory << "\n";
    cout << "Async engines              : " << p.asyncEngineCount << "\n";
}

void printLowLevel(int dev)
{
    cout << "\n============== LOW-LEVEL ARCHITECTURE ==============\n";

    cout << "Max threads per SM             : "
              << getAttr(cudaDevAttrMaxThreadsPerMultiProcessor, dev) << "\n";

    cout << "Max blocks per SM              : "
              << getAttr(cudaDevAttrMaxBlocksPerMultiprocessor, dev) << "\n";

    // Computed dynamically as (MaxThreadsPerMultiProcessor / WarpSize) 
    // because cudaDevAttrMaxWarpsPerMultiprocessor is not a native CUDA enum
    int maxThreads = getAttr(cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    int warpSize = getAttr(cudaDevAttrWarpSize, dev);
    cout << "Max warps per SM               : "
              << (maxThreads / warpSize) << "\n";

    cout << "Registers per SM               : "
              << getAttr(cudaDevAttrMaxRegistersPerMultiprocessor, dev) << "\n";


    cout << "Shared mem per SM              : "
              << getAttr(cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev)
              << " B\n";

    cout << "Shared mem per block (opt-in)  : "
              << getAttr(cudaDevAttrMaxSharedMemoryPerBlockOptin, dev)
              << " B\n";

    cout << "Runtime shared mem overhead    : "
              << getAttr(cudaDevAttrReservedSharedMemoryPerBlock, dev)
              << " B\n";

    cout << "Multiprocessor count           : "
              << getAttr(cudaDevAttrMultiProcessorCount, dev) << "\n";

    cout << "\n[Tensor Core Hints]\n";
    if (getAttr(cudaDevAttrComputeCapabilityMajor, dev) >= 7)
        cout << "Tensor cores supported         : YES\n";
    else
        cout << "Tensor cores supported         : NO\n";

    if (getAttr(cudaDevAttrComputeCapabilityMajor, dev) >= 8)
        cout << "TF32 supported                 : YES\n";
    else
        cout << "TF32 supported                 : NO\n";
}

int main()
{
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        cout << "No CUDA devices found.\n";
        return 0;
    }

    cout << "Detected " << deviceCount << " CUDA device(s)\n";

    for (int dev = 0; dev < deviceCount; dev++) {

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        cout << "\n====================================================\n";
        cout << "DEVICE " << dev << "\n";
        cout << "====================================================\n";

        printBasic(prop);
        printLowLevel(dev);
    }

    return 0;
}
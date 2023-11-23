// This program computes the sum of two vectors of length N using pinned memory
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <thread>

using std::begin;
using std::copy;
using std::end;
using std::generate;
using std::vector;

#define cuda_try(call)                                                                \
  do {                                                                                \
    cudaError_t err = static_cast<cudaError_t>(call);                                 \
    if (err != cudaSuccess) {                                                         \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorName(err)); \
      std::terminate();                                                               \
    }                                                                                 \
  } while (0)

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Boundary check
  if (tid < N) c[tid] += a[tid] + b[tid];
}

// Check vector add result
void verify_result(std::vector<int> &a, std::vector<int> &b,
                   std::vector<int> &c) {
  for (int i = 0; i < a.size(); i++) {
    assert(c[i] == a[i] + b[i]);
  }
}


CUdevice get_cuda_device(const int device_id, int& sms_count) {
  CUdevice device;
  int device_count = 0;

  cuda_try(cuInit(0));  // Flag parameter must be zero
  cuda_try(cuDeviceGetCount(&device_count));

  if (device_count == 0) {
    std::cout << "No CUDA capable device found." << std::endl;
    std::terminate();
  }

  cuda_try(cuDeviceGet(&device, device_id));

  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);

  sms_count = device_prop.multiProcessorCount;

  std::cout << "Device[" << device_id << "]: " << device_prop.name << '\n';
  std::cout << "SMs count: " << sms_count << '\n';
  std::cout << "Total amount of global memory: " << device_prop.totalGlobalMem << ", " << device_prop.totalGlobalMem / 1048576.0f << " MBytes \n";

  return device;
}

int main() {
  int device_id = 0;
  int sms_count = 0;
  CUdevice dev = get_cuda_device(device_id, sms_count);

  //CUcontext ctx0, ctx1;

  //cuCtxCreate(&ctx0, CUctx_flags::CU_CTX_SCHED_AUTO, dev);
  //cuCtxCreate(&ctx1, CUctx_flags::CU_CTX_SCHED_AUTO, dev);

  /*CUcontext victimContext;

  cuCtxCreate(&victimContext, CUctx_flags::CU_CTX_SCHED_AUTO, dev);

  cuCtxSetCurrent(victimContext);*/

  /*size_t total_memsize, available_memsize;
  cudaMemGetInfo(&available_memsize, &total_memsize);
  size_t victim_base_usage = total_memsize - available_memsize;
  std::cout << "Available Memory: " << available_memsize << ", " << available_memsize / 1048576.0f << " MBytes\n"; 
  std::cout << "Victim base memory usage: " << victim_base_usage << ", " << victim_base_usage / 1048576.0f << " MBytes \n";*/

  // Array size of 2^16 (65536 elements)
  constexpr int N = 1 << 26;
  size_t bytes = sizeof(int) * N;

  // Vectors for holding the host-side (CPU-side) data
  std::vector<int> h_a;
  h_a.reserve(N);
  std::vector<int> h_b;
  h_b.reserve(N);
  std::vector<int> h_c;
  h_c.reserve(N);

  // Initialize random numbers in each array
  for (int i = 0; i < N; i++) {
    h_a.push_back(rand() % 100);
    h_b.push_back(rand() % 100);
  }

  // Allocate memory on the device
  int *d_a, *d_b, *d_c;
  cuda_try(cudaMalloc(&d_a, bytes));
  cuda_try(cudaMalloc(&d_b, bytes));
  cuda_try(cudaMalloc(&d_c, bytes));

  size_t total_memsize, available_memsize;
  cudaMemGetInfo(&available_memsize, &total_memsize);
  size_t victim_base_usage = total_memsize - available_memsize;
  std::cout << "Victim base memory usage: " << victim_base_usage << ", " << victim_base_usage / 1048576.0f << " MBytes \n";

  // Copy data from the host to the device (CPU -> GPU)
  cuda_try(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  cuda_try(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  // Threads per CTA (1024 threads per CTA)
  int NUM_THREADS = 1 << 10;

  // CTAs per Grid
  // We need to launch at LEAST as many threads as we have elements
  // This equation pads an extra CTA to the grid if N cannot evenly be divided
  // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  // Launch the kernel on the GPU
  // Kernel calls are asynchronous (the CPU program continues execution after
  // call, but no necessarily before the kernel finishes)

  //std::thread thread0{[&] {
    //cuCtxSetCurrent(ctx0);
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
  //}};

  /*std::thread thread1{[&] {
    //cuCtxSetCurrent(ctx1);
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_c, d_d, N);
  }};*/

  //thread0.join();
  //thread1.join();

  cuda_try(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  // Check result for errors
  verify_result(h_a, h_b, h_c);

  std::cout << "Output Verified\n";

  // Free memory on device
  cuda_try(cudaFree(d_a));
  cuda_try(cudaFree(d_b));
  cuda_try(cudaFree(d_c));

  //cuda_try(cuCtxDestroy(ctx0));
  //cuda_try(cuCtxDestroy(ctx1));
  //cuda_try(cuCtxDestroy(victimContext));

  /*std::cout << "Dumping out GPU memory\n";

  cuda_try(cudaMalloc(&d_a, bytes));
  cuda_try(cudaMalloc(&d_b, bytes));
  cuda_try(cudaMalloc(&d_c, bytes));

  std::vector<int> h_d;
  h_d.reserve(N);
  std::vector<int> h_e;
  h_e.reserve(N);
  std::vector<int> h_f;
  h_f.reserve(N);

  cuda_try(cudaMemcpy(h_d.data(), d_a, bytes, cudaMemcpyDeviceToHost));
  cuda_try(cudaMemcpy(h_e.data(), d_b, bytes, cudaMemcpyDeviceToHost));
  cuda_try(cudaMemcpy(h_f.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    if(h_d[i] != 0) {
      std::cout << "Non zero element! h_d[" << i << "] = " << h_d[i] << "\n"; 
    }
    if(h_e[i] != 0) {
      std::cout << "Non zero element! h_e[" << i << "] = " << h_e[i] << "\n"; 
    }
    if(h_f[i] != 0) {
      std::cout << "Non zero element! h_f[" << i << "] = " << h_f[i] << "\n"; 
    }
  }*/

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
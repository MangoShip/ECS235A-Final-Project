
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <thread>

#include <unistd.h>

#define cuda_try(call)                                                                \
  do {                                                                                \
    cudaError_t err = static_cast<cudaError_t>(call);                                 \
    if (err != cudaSuccess) {                                                         \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorName(err)); \
      std::terminate();                                                               \
    }                                                                                 \
  } while (0)

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


__global__ void transferMem(uint32_t *d_a, size_t num_elements) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < num_elements) {
    d_a[tid]++;
  }
  
}

// a 1024-bit random sequence
unsigned int uniq_key[32] = {
        0x63636363U, 0x7c7c7c7cU, 0x77777777U, 0x7b7b7b7bU,
        0xf2f2f2f2U, 0x6b6b6b6bU, 0x6f6f6f6fU, 0xc5c5c5c5U,
        0x30303030U, 0x01010101U, 0x67676767U, 0x2b2b2b2bU,
        0xfefefefeU, 0xd7d7d7d7U, 0xababababU, 0x76767676U,
        0x239c9cbfU, 0x53a4a4f7U, 0xe4727296U, 0x9bc0c05bU,
        0x75b7b7c2U, 0xe1fdfd1cU, 0x3d9393aeU, 0x4c26266aU,
        0x6c36365aU, 0x7e3f3f41U, 0xf5f7f702U, 0x83cccc4fU,
        0x6834345cU, 0x51a5a5f4U, 0xd1e5e534U, 0xf9f1f108U
};

int main() {
  int device_id = 0;
  int sms_count = 0;
  CUdevice dev = get_cuda_device(device_id, sms_count);

  size_t total_memsize, available_memsize;

  cudaMemGetInfo(&available_memsize, &total_memsize);
  size_t attacker_base_usage = total_memsize - available_memsize;
  std::cout << "Attacker base memory usage: " << attacker_base_usage << ", " << attacker_base_usage / 1048576.0f << " MBytes \n";
  std::cout << "Available Memory: " << available_memsize << ", " << available_memsize / 1048576.0f << " MBytes\n"; 

  size_t memsize = available_memsize - 65273856;

  /*uint32_t *a;
  a = (uint32_t*)malloc(memsize);*/

  size_t num_elements = memsize / 4;
  std::cout << "Number of elements in a: " << num_elements << "\n";
  /*for (size_t i = 0; i < memsize/4; i += 32) {
		for (size_t j = 0; j < 32; ++j) {
			a[i+j] = uniq_key[j];
		}
	} */

  uint32_t *a;
  cuda_try(cudaMallocManaged(&a, memsize));
  std::cout << "Allocating: " << memsize << ", " << memsize / 1048576.0f << " MBytes\n"; 
  //cuda_try(cudaMemcpy(d_a, a, memsize, cudaMemcpyHostToDevice)); // fill GPU memory with a predefined value
  //cuda_try(cudaFree(d_a)); // deallocate it!!

  for (size_t i = 0; i < memsize/4; i += 32) {
		for (size_t j = 0; j < 32; ++j) {
			a[i+j] = uniq_key[j];
		}
	} 

  int num_threads = 1024;
  size_t num_blocks = num_elements / num_threads;

  std::cout << "Launching GPU kernel (transferMem)\n";
  transferMem<<<num_blocks, num_threads>>>(a, num_elements);

  cudaDeviceSynchronize();
  //cudaFree(&a);
  cudaMemGetInfo(&available_memsize, &total_memsize);
  std::cout << "Available Memory: " << available_memsize << ", " << available_memsize / 1048576.0f << " MBytes\n"; 

  unsigned int *d_a;
  cuda_try(cudaMalloc(&d_a, memsize));
  cudaMemGetInfo(&available_memsize, &total_memsize);
  std::cout << "Available Memory: " << available_memsize << ", " << available_memsize / 1048576.0f << " MBytes\n"; 

  uint32_t *h_a;
  h_a = (uint32_t*)malloc(memsize);
  cuda_try(cudaMemcpy(h_a, d_a, memsize, cudaMemcpyDeviceToHost));

  cudaMemGetInfo(&available_memsize, &total_memsize);
  std::cout << "Available Memory: " << available_memsize << ", " << available_memsize / 1048576.0f << " MBytes\n"; 

  /*uint32_t *h_a;
  cuda_try(cudaMallocManaged(&h_a, memsize));
  cudaMemPrefetchAsync(&h_a, memsize, cudaCpuDeviceId);

  size_t num_uncleared = 0;
  size_t num_zero = 0;
  size_t num_others = 0;

  for (size_t i = 0; i < memsize / 4; i+= 32) {
    for (size_t j = 0; j < 32; ++j) {
			if(h_a[i+j] == uniq_key[j] + 1) {
        num_uncleared++;
      }
      else if(h_a[i+j] == 0) {
        num_zero++;
      }
      else {
        num_others++;
      }
		}
  }

    std::cout << "Number of uncleared memory: " << num_uncleared << "\n";
  std::cout << "Number of zeros: " << num_zero << "\n";
  std::cout << "Number of others: " << num_others << "\n";*/
}
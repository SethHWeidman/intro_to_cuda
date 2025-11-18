#include <cmath>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define THREADS_PER_BLK 128

// 1-D convolution with a width-3 box filter computed directly from global memory.
__global__ void convolve(int N, float *input, float *output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x; // thread local variable
  float result = 0.0f;                               // thread-local variable

  for (int i = 0; i < 3; i++)
    result += input[index + i];

  output[index] = result / 3.f;
}

// Same convolution, but each block cooperatively loads a tile + halo into shared
// memory so each input element is read from global memory once.
__global__ void convolve_shared_memory(int N, float *input, float *output) {
  // Each block stages THREADS_PER_BLK elements plus a 2-value halo so the 3-wide
  // window fits even for the last two threads in the block.
  __shared__ float support[THREADS_PER_BLK + 2];     // per-block staging buffer
  int index = blockIdx.x * blockDim.x + threadIdx.x; // thread local variable

  // Example: thread 126 copies input[254] into support[126] so every thread stages its own value.
  support[threadIdx.x] = input[index];
  // Only threads 126 and 127 satisfy this check, so they fetch the halo values beyond the tile.
  if (threadIdx.x >= THREADS_PER_BLK - 2) {
    // haloIdx tells whether the thread handles the first or second halo slot (126 -> 0, 127 -> 1).
    int haloIdx = threadIdx.x - (THREADS_PER_BLK - 2);
    // blockStart rewinds to the block's first element (254 - 126 = 128) so indexing stays local to
    // the block.
    int blockStart = index - threadIdx.x;
    // The halo value lives right after the tile (128 + 128 + haloIdx = 256/257), so write it into
    // support[128/129].
    support[THREADS_PER_BLK + haloIdx] = input[blockStart + THREADS_PER_BLK + haloIdx];
  }

  __syncthreads(); // make sure the whole tile/halo is visible to every thread

  float result = 0.0f; // thread-local variable
  for (int i = 0; i < 3; i++)
    result += support[threadIdx.x + i];

  output[index] = result / 3.f;
}

void my_launcher() {
  const int N = 1 << 20; // tune N to make the timing signal larger/smaller
  float *input = new float[N + 2];
  for (int i = 0; i < N + 2; i++)
    input[i] = i % 2;

  int threadsPerBlock = THREADS_PER_BLK;
  int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  float *devInput = nullptr;
  float *devOutput = nullptr;

  cudaMalloc(&devInput, sizeof(float) * (N + 2)); // allocate input array in device memory
  cudaMalloc(&devOutput, sizeof(float) * N);      // allocate output array in device memory

  cudaMemcpy(devInput, input, (N + 2) * sizeof(float), cudaMemcpyHostToDevice);

  // Host buffer holding the naive kernel's output so we can compare later. Needs to be
  // heap-allocated (pointer) because N is far too large for the stack.
  float *globalMemOutput = new float[N];
  // Host buffer holding the shared-memory kernel's output for verification/printing. Same reasoning
  // for using a pointer as above.
  float *sharedMemOutput = new float[N];

  convolve<<<numBlocks, threadsPerBlock>>>(N, devInput, devOutput);
  cudaDeviceSynchronize();
  cudaMemcpy(globalMemOutput, devOutput, N * sizeof(float), cudaMemcpyDeviceToHost);

  convolve_shared_memory<<<numBlocks, threadsPerBlock>>>(N, devInput, devOutput);
  cudaDeviceSynchronize();
  cudaMemcpy(sharedMemOutput, devOutput, N * sizeof(float), cudaMemcpyDeviceToHost);

  float maxDiff = 0.0f;
  for (int i = 0; i < N; i++) {
    float diff = fabsf(globalMemOutput[i] - sharedMemOutput[i]);
    if (diff > maxDiff)
      maxDiff = diff;
  }

  printf("Shared memory result matches naive implementation? %s (max abs diff %.6f)\n",
         maxDiff < 1e-6f ? "yes" : "no", maxDiff);
  printf("Sample outputs (shared-memory kernel): ");
  for (int i = 0; i < 8 && i < N; i++)
    printf("%.1f ", sharedMemOutput[i]);
  printf("...\n");

  delete[] input;
  delete[] globalMemOutput;
  delete[] sharedMemOutput;
  cudaFree(devInput);
  cudaFree(devOutput);
}

void printCudaInfo() {

  // print out stats about the GPU in the machine.  Useful if
  // students want to know what GPU they are running on.

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
           static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
}

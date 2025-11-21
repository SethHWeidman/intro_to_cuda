#pragma once

#include <cmath>
#include <cstddef>

#ifndef BLOCK_DIM
#define BLOCK_DIM 32
#endif

template <typename T>
__global__ void mm_kernel(T const *A, T const *B, T *C, size_t M, size_t K, size_t N) {
  size_t i{blockIdx.y * blockDim.y + threadIdx.y};
  size_t j{blockIdx.x * blockDim.x + threadIdx.x};

  if ((i >= M) || (j >= N)) {
    return;
  }

  T acc_sum{0};
  for (size_t k{0}; k < K; ++k) {
    acc_sum += A[i * K + k] * B[k * N + j];
  }
  C[i * N + j] = acc_sum;
}

template <typename T>
__global__ void mm_kernel_shared_memory(T const *A, T const *B, T *C, size_t M, size_t K,
                                        size_t N) {
  __shared__ T A_tile[BLOCK_DIM][BLOCK_DIM];
  __shared__ T B_tile[BLOCK_DIM][BLOCK_DIM];

  T acc_sum{0};

  for (size_t tile_idx{0}; tile_idx < ceilf(static_cast<float>(K) / BLOCK_DIM); ++tile_idx) {
    size_t i{blockIdx.y * blockDim.y + threadIdx.y};
    size_t j{tile_idx * blockDim.x + threadIdx.x};
    if ((i < M) && (j < K)) {
      A_tile[threadIdx.y][threadIdx.x] = A[i * K + j];
    } else {
      A_tile[threadIdx.y][threadIdx.x] = 0;
    }

    i = tile_idx * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < K) && (j < N)) {
      B_tile[threadIdx.y][threadIdx.x] = B[i * N + j];
    } else {
      B_tile[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    for (size_t k{0}; k < BLOCK_DIM; ++k) {
      acc_sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
    }
    __syncthreads();
  }

  size_t i{blockIdx.y * blockDim.y + threadIdx.y};
  size_t j{blockIdx.x * blockDim.x + threadIdx.x};

  if ((i < M) && (j < N)) {
    C[i * N + j] = acc_sum;
  }
}

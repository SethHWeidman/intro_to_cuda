// CODE ADAPTED FROM:
// https://leimao.github.io/blog/CUDA-Matrix-Multiplication/#Matrix-Multiplication-Optimizations
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "CycleTimer.h"
#include "matmul_kernels.cuh"

/******************************************************************************/
/*                               Utilities                                    */
/******************************************************************************/
#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <typename T> std::vector<T> create_rand_vector(size_t K) {
  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_int_distribution<int> uniform_dist(-256, 256);

  std::vector<T> vec(K);
  for (size_t i{0}; i < K; ++i) {
    vec.at(i) = static_cast<T>(uniform_dist(e));
  }

  return vec;
}

template <typename T>
bool allclose(std::vector<T> const &vec_1, std::vector<T> const &vec_2, T const &abs_tol) {
  if (vec_1.size() != vec_2.size()) {
    return false;
  }
  for (size_t i{0}; i < vec_1.size(); ++i) {
    if (std::abs(vec_1.at(i) - vec_2.at(i)) > abs_tol) {
      printf("Elements and index %u do not match: (%f, %f)\n", unsigned(i), vec_1.at(i),
             vec_2.at(i));
      return false;
    }
  }
  return true;
}

/******************************************************************************/
/*                              Main Functions                                */
/******************************************************************************/


// Computes C = A * B on the CPU
// A: M x K
// B: K x N
// C: M x N
template <typename T> void mm(T const *A, T const *B, T *C, size_t M, size_t K, size_t N) {
  // Compute the cells in C sequentially.
  for (size_t i{0}; i < M; ++i) {
    for (size_t j{0}; j < N; ++j) {
      T acc_sum{0};
      for (size_t k{0}; k < K; ++k) {
        acc_sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc_sum;
    }
  }
}

/******************************************************************************/
/*                                  Driver                                    */
/******************************************************************************/

int main() {
  struct TimingEntry {
    std::string dataset;
    std::string implementation;
    double seconds;
  };
  enum class ExperimentKind { CpuVsGpu, GpuComparison };

  auto printTimingTable = [](std::string const &title, std::vector<TimingEntry> const &entries) {
    std::cout << "\n" << title << std::endl;
    printf("%-28s %12s\n", "Implementation", "Time");
    for (auto const &entry : entries) {
      printf("%-28s %12.3f\n", entry.implementation.c_str(), entry.seconds);
    }
  };

  auto runGpuKernel = [&](auto launchKernel, const char *label, float *d_C_GPU,
                          std::vector<float> &host_output) -> double {
    checkCuda(cudaMemset(d_C_GPU, 0, sizeof(float) * host_output.size()));

    double startTime = CycleTimer::currentSeconds();
    launchKernel();
    checkCuda(cudaDeviceSynchronize());
    double endTime = CycleTimer::currentSeconds();

    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess) {
      std::cerr << "CUDA Matrix Multiplication kernel failed to execute (" << label << ")."
                << std::endl;
      std::cerr << cudaGetErrorString(err) << std::endl;
      std::exit(EXIT_FAILURE);
    }

    checkCuda(cudaMemcpy(host_output.data(), d_C_GPU, sizeof(float) * host_output.size(),
                         cudaMemcpyDeviceToHost));
    return endTime - startTime;
  };

  auto runExperiment = [&](size_t dim, ExperimentKind kind) {
    const size_t M{dim}, K{dim}, N{dim};
    std::string datasetLabel = std::to_string(dim) + " x " + std::to_string(dim);
    std::cout << "\n=== Running experiment for " << datasetLabel << " ===" << std::endl;

    bool includeCpu = (kind == ExperimentKind::CpuVsGpu);
    bool includeShared = (kind == ExperimentKind::GpuComparison);

    std::vector<float> const A_vec{create_rand_vector<float>(M * K)};
    std::vector<float> const B_vec{create_rand_vector<float>(K * N)};
    std::vector<float> C_GPU_naive_vec(M * N);
    std::vector<float> C_GPU_shared_vec;
    if (includeShared) {
      C_GPU_shared_vec.resize(M * N);
    }
    std::vector<float> C_CPU_vec;
    float *C_CPU{nullptr};
    if (includeCpu) {
      C_CPU_vec.resize(M * N);
      C_CPU = C_CPU_vec.data();
    }

    float const *A{A_vec.data()};
    float const *B{B_vec.data()};

    float *d_A{nullptr}, *d_B{nullptr}, *d_C_GPU{nullptr};
    checkCuda(cudaMalloc(&d_A, sizeof(float) * A_vec.size()));
    checkCuda(cudaMalloc(&d_B, sizeof(float) * B_vec.size()));
    checkCuda(cudaMalloc(&d_C_GPU, sizeof(float) * C_GPU_naive_vec.size()));

    checkCuda(cudaMemcpy(d_A, A, sizeof(float) * A_vec.size(), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, B, sizeof(float) * B_vec.size(), cudaMemcpyHostToDevice));

    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(N) / threads_per_block.x);
    blocks_per_grid.y = std::ceil(static_cast<double>(M) / threads_per_block.y);

    std::vector<TimingEntry> dataset_timings;

    std::cout << "Running GPU kernel" << (includeShared ? "s" : "") << "..." << std::endl;
    double naive_time = runGpuKernel(
        [&] { mm_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C_GPU, M, K, N); },
        "GPU (naive global memory)", d_C_GPU, C_GPU_naive_vec);
    dataset_timings.push_back({datasetLabel, "GPU (naive global memory)", naive_time});

    if (includeShared) {
      double shared_time = runGpuKernel(
          [&] {
            mm_kernel_shared_memory<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C_GPU, M, K,
                                                                            N);
          },
          "GPU (shared memory tile)", d_C_GPU, C_GPU_shared_vec);
      dataset_timings.push_back({datasetLabel, "GPU (shared memory tile)", shared_time});
    }

    if (includeCpu) {
      std::cout << "Running CPU matmul..." << std::endl;
      double cpuStart = CycleTimer::currentSeconds();
      mm<float>(A, B, C_CPU, M, K, N);
      double cpuEnd = CycleTimer::currentSeconds();
      dataset_timings.push_back({datasetLabel, "CPU (single-thread)", cpuEnd - cpuStart});

      bool naive_match = allclose<float>(C_CPU_vec, C_GPU_naive_vec, 1e-4);
      std::cout << "Numeric results check (CPU vs GPU naive): "
                << (naive_match ? "match" : "DIFFER") << std::endl;
    }
    if (includeShared) {
      bool gpu_match = allclose<float>(C_GPU_naive_vec, C_GPU_shared_vec, 1e-4);
      std::cout << "Numeric results check (GPU naive vs GPU shared): "
                << (gpu_match ? "match" : "DIFFER") << std::endl;
    }

    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C_GPU));

    return dataset_timings;
  };

  auto smallTimings = runExperiment(1024, ExperimentKind::CpuVsGpu);
  printTimingTable("Timing summary for experiment", smallTimings);

  auto largeTimings = runExperiment(8192, ExperimentKind::GpuComparison);
  printTimingTable("Timing summary for experiment", largeTimings);

  return 0;
}

# Demo 3 – Tiled Matrix Multiply

This demo benchmarks three implementations of square matrix multiplication:

- A CPU baseline (`mm`) that runs on a single host core.
- A naive GPU kernel that reads every input element directly from global memory.
- A tiled GPU kernel that cooperatively stages blocks of `A` and `B` into `__shared__` memory.

## Prerequisites

- CUDA-capable GPU with a driver/toolkit that provides `nvcc` and the CUDA runtime.
- `make`, `g++`, and permission to compile inside this folder.

## Build

1. `cd intro_to_cuda/demo3_matmul`
2. Remove any previous binary: `make clean`
3. Build the executable: `make`

> **Note:** the `-gencode arch=compute_89,code=sm_89` `NVCCFLAGS` argument tells the code to
> generate PTX and machine code for the Ada Lovelace / L4 GPUs which NVIDIA lists as "Compute
> Capability 8.9". See [NVIDIA's CUDA GPU list](https://developer.nvidia.com/cuda-gpus). Modify
> these flags if using a GPU other than an L4.

## Run

After a successful build, run:
```bash
./matmul
```
or, equivalently:
```bash
make run
```

### What you'll see

- A small 1024×1024 experiment that compares the timing of the CPU and naive GPU implementations as
  well as checking their results for numerical agreement
- A larger 8192×8192 experiment that compares the timing of the naive GPU kernel against the
  shared-memory tiled kernel, again verifying that both produce the same output.

### Sample output

```text
=== Running experiment for 1024 x 1024 ===
Running GPU kernel...
Running CPU matmul...
Numeric results check (CPU vs GPU naive): match

Timing summary for experiment
Implementation                       Time
GPU (naive global memory)           0.001
CPU (single-thread)                 2.742

=== Running experiment for 8192 x 8192 ===
Running GPU kernels...
Numeric results check (GPU naive vs GPU shared): match

Timing summary for experiment
Implementation                       Time
GPU (naive global memory)           0.900
GPU (shared memory tile)            0.553
```

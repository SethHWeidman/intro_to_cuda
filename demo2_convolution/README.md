# Demo 2 – Shared-Memory Convolution

This example benchmarks two 1-D convolution kernels: a naive implementation that reads every input
directly from global memory, and a shared-memory version that declares a block-scoped `__shared__
float support[]` buffer to reduce reads and writes from global memory / DRAM.

## Prerequisites
- CUDA-capable GPU with a toolkit/driver that provides `nvcc` and the CUDA runtime.
- `make`, `g++`, and permission to compile inside this folder.

## Build
1. `cd intro_to_cuda/demo2_convolution`
2. Remove any stale objects: `make clean`
3. Build the executable: `make`. The Makefile defaults to `-gencode arch=compute_89,code=sm_89`
   (optimized for NVIDIA L4). If your GPU uses a different compute capability, edit `NVCCFLAGS` or
   pass your own flags:
   ```
   make NVCCFLAGS="-gencode arch=compute_<XY>,code=sm_<XY>"
   ```

## Run
After a successful build:
```
./demo2
```

### What you'll see
The program launches both kernels on one million elements and confirms that they produce the same
result.

### Sample output
```
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA L4
   SMs:        58
   Global mem: 22478 MB
   CUDA Cap:   8.9
---------------------------------------------------------

Shared memory result matches naive implementation? yes (max abs diff 0.000000)
Sample outputs (shared-memory kernel): 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 ...
```

## Shared memory cuts bandwidth by roughly 2x

TL;DR, we go from reading each piece of data three times to reading it once, but still write
everything once.

In the naive kernel every output element triggers four global-memory transactions: three reads for
the sliding 3-wide window plus one write for the result. With `N` outputs (and `sizeof(float)` bytes
per element) that becomes `4 * N * sizeof(float)` bytes moving between DRAM and the SMs. When the
thread block cooperatively stages its tile in `__shared__` memory, the block pulls in
`THREADS_PER_BLK + 2` floats once per each of `numBlocks` blocks, but still writes a total of `N`
outputs back. That corresponds to `((THREADS_PER_BLK + 2) * numBlocks + N) * sizeof(float)`. Because
`(THREADS_PER_BLK + 2) * numBlocks ≈ THREADS_PER_BLK * numBlocks = N`, there is roughly `(N + N) *
sizeof(float) = 2 * N * sizeof(float)` total reading and writing, half of the implementation that
does not use shared memory.
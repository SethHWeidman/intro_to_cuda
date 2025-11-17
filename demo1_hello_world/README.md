# Demo 1 – CUDA Hello World

This sample prints the CUDA device properties and, when supported, the thread/block coordinates from
`my_kernel_2D`. Follow the steps below to rebuild it with a recent toolkit and capture the output.

## Prerequisites
- CUDA-capable GPU with a driver/toolkit that provides `nvcc` and `cuda-runtime`.
- `make`, `g++`, and write access to this folder.

## Build
1. `cd intro_to_cuda/demo1_hello_world`
2. Clean any prior artifacts: `make clean`
3. Build the executable: `make`. The Makefile now targets `--gpu-architecture compute_89`, which
   matches the NVIDIA L4 GPU used for these demos. If you are on different hardware, edit the
   `NVCCFLAGS` line in the Makefile (or pass your own via `make NVCCFLAGS="..."`) to match your
   device’s compute capability—see NVIDIA’s [CUDA GPU list](https://developer.nvidia.com/cuda-gpus)
   for the right values.

## Run
Execute the binary after a successful build:
```
./demo1
```

### Sample output
```
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA L4
   SMs:        58
   Global mem: 22478 MB
   CUDA Cap:   8.9
---------------------------------------------------------
rowID 0 colID 0 blockDim.x 4, blockDim.y 3 gridDim.x 1 gridDim.y 1 numRows 3 numCols 4 myThreadID 0
rowID 0 colID 1 blockDim.x 4, blockDim.y 3 gridDim.x 1 gridDim.y 1 numRows 3 numCols 4 myThreadID 1
rowID 0 colID 2 blockDim.x 4, blockDim.y 3 gridDim.x 1 gridDim.y 1 numRows 3 numCols 4 myThreadID 2
rowID 0 colID 3 blockDim.x 4, blockDim.y 3 gridDim.x 1 gridDim.y 1 numRows 3 numCols 4 myThreadID 3
rowID 1 colID 0 blockDim.x 4, blockDim.y 3 gridDim.x 1 gridDim.y 1 numRows 3 numCols 4 myThreadID 4
rowID 1 colID 1 blockDim.x 4, blockDim.y 3 gridDim.x 1 gridDim.y 1 numRows 3 numCols 4 myThreadID 5
rowID 1 colID 2 blockDim.x 4, blockDim.y 3 gridDim.x 1 gridDim.y 1 numRows 3 numCols 4 myThreadID 6
rowID 1 colID 3 blockDim.x 4, blockDim.y 3 gridDim.x 1 gridDim.y 1 numRows 3 numCols 4 myThreadID 7
rowID 2 colID 0 blockDim.x 4, blockDim.y 3 gridDim.x 1 gridDim.y 1 numRows 3 numCols 4 myThreadID 8
rowID 2 colID 1 blockDim.x 4, blockDim.y 3 gridDim.x 1 gridDim.y 1 numRows 3 numCols 4 myThreadID 9
rowID 2 colID 2 blockDim.x 4, blockDim.y 3 gridDim.x 1 gridDim.y 1 numRows 3 numCols 4 myThreadID 10
rowID 2 colID 3 blockDim.x 4, blockDim.y 3 gridDim.x 1 gridDim.y 1 numRows 3 numCols 4 myThreadID 11
```

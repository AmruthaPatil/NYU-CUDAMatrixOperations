# CUDA Matrix Operations

## Introduction
This project focuses on optimizing CUDA programs for vector addition and matrix multiplication. It demonstrates how modifying memory access patterns and parallelism can enhance the performance of these operations on NVIDIA GPUs. The project involves adjusting CUDA kernels for better memory coalescence and exploring various optimization techniques for matrix multiplication.

## Methodology
The project comprises two main tasks:

### Vector Addition (Non-Coalesced and Coalesced):
- **Non-Coalesced Vector Addition (vecadd00):** The base version performs vector addition without coalesced memory reads, utilizing `vecaddKernel00.cu`.
- **Coalesced Vector Addition (vecadd01):** An optimized version that implements coalesced memory access to improve performance, utilizing `vecaddKernel01.cu`.

### Matrix Multiplication:
- **Shared CUDA Matrix Multiply (matmult00):** Executes matrix multiplication using basic CUDA without special optimizations.
- **Improved Shared CUDA Matrix Multiply (matmult01):** Enhances the `matmult00` by optimizing memory access patterns and possibly implementing tiling techniques.

## Observations
### Vector Addition
- **Non-Coalesced vs. Coalesced:** Coalesced vector addition significantly improves performance metrics (execution time and throughput) due to better memory access alignment with the GPU's architecture.
### Matrix Multiplication
- **Performance Enhancement:** The optimized `matmult01` showed a notable reduction in execution time and increase in GFlops/s across different matrix sizes compared to `matmult00`.
- Advanced CUDA optimizations like tiling and memory coalescing were instrumental in achieving these gains.

## How to Run the Code
Ensure CUDA and the appropriate NVIDIA drivers are installed. Use the following commands in a terminal:

- **To compile the code:**
```bash
  cd path/to/folder
  make clean
  make 
```

- **To run the vector addition:**
```bash
./vecadd00 [arguments]  # For non-coalesced
./vecadd01 [arguments]  # For coalesced 
```

- **To run the matrix multiplication:**
```bash
./matmul00 [arguments]  # For basic multiplication
./matmul01 [arguments]  # For optimized multiplication
```
Replace [arguments] with the appropriate values as per the program requirements.

Conclusion
This assignment illustrates the crucial role of memory access patterns and parallel computing optimizations in enhancing the performance of CUDA applications. The experiments conducted provide valuable insights into the scalability and efficiency of GPU programming.
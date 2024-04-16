#include "matmultKernel.h"

// CUDA kernel to perform matrix multiplication on the GPU.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
    // Compute offsets in the output matrix for coalesced memory access.
    int outputRowOffset = BLOCK_SIZE * B.width / FOOTPRINT_SIZE;
    int outputColOffset = BLOCK_SIZE * A.height / FOOTPRINT_SIZE;

    // Calculate row and column indices for the current thread within the output matrix.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Variables to accumulate the products for the output matrix C.
    float sumForC1 = 0, sumForC2 = 0, sumForC3 = 0, sumForC4 = 0;

    // Loop over all elements in the row of A and column of B to compute product contributions to C.
    for (int k = 0; k < A.width; ++k) {
        sumForC1 += A.elements[row * A.width + k] * B.elements[k * B.width + col];
        sumForC2 += A.elements[(row + outputRowOffset) * A.width + k] * B.elements[k * B.width + col];
        sumForC3 += A.elements[row * A.width + k] * B.elements[k * B.width + (col + outputColOffset)];
        sumForC4 += A.elements[(row + outputRowOffset) * A.width + k] * B.elements[k * B.width + (col + outputColOffset)];
    }   
    // Store the results in the appropriate elements of matrix C.
    C.elements[row * C.width + col] = sumForC1;
    C.elements[(row + outputRowOffset) * C.width + col] = sumForC2;
    C.elements[row * C.width + (col + outputColOffset)] = sumForC3;
    C.elements[(row + outputRowOffset) * C.width + (col + outputColOffset)] = sumForC4;
}

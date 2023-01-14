// System includes
#include <stdio.h>
#include <assert.h>
#include <chrono>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "matrixOperations.cuh"

__global__ void hello(){
    printf("hello\n");
}

int main(){
	/*printf("[Matrix Multiply Using CUDA] - Starting...\n");

  

  dim3 dimsA(5 * 2 * block_size, 16, 1);
  dim3 dimsB(16, 5 * 2 * block_size, 1);

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }



  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
         dimsB.x, dimsB.y);

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(block_size, dimsA, dimsB);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);*/
  int block_size = 32;

	dim3 dimsA(5 * 2 * block_size, 5*4*block_size, 1);
  dim3 dimsB(5*4*block_size, 5 * block_size, 1);
  
    dim3 threads(block_size,block_size,1);
  dim3 blocks((dimsB.y + block_size - 1) / block_size, (dimsA.x + block_size - 1) / block_size);

  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(double) * size_A;

  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(double) * size_B;
  
  unsigned int size_C = dimsA.x * dimsB.y;
  unsigned int mem_size_C = sizeof(double) * size_C;
  
    // Allocate host memory
  double *h_A, *h_B, *h_C;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));
  
  // Initialize host memory matrices A and B
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, 5.0f);


// Allocate device memory
  double *d_A, *d_B, *d_C;

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
  
    // copy host memory to device
  checkCudaErrors(
      cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
      
    matMul<<<blocks,threads>>>(d_A,d_B,d_C,dimsA.x,dimsA.y,dimsB.y);
    cudaDeviceSynchronize();
    
      // Copy result from device to host
  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaDeviceSynchronize());
    
    printf("%f\n",h_C[59]);
      
      
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));
	checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

}

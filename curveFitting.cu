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

//Generating input data
#include "data.h"

//matrix operation using GPU
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
  dim3 threads(block_size,block_size,1);
  
  int nSamples=10;
  int order=6;
  
  //Allocate host x
  unsigned int mem_size_x = sizeof(double) * nSamples;
  double *h_x;
  checkCudaErrors(cudaMallocHost(&h_x, mem_size_x));
  
  //Allocate device x
  double *d_x;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_x), mem_size_x));
  /*
  //constant init x
  ConstantInit(h_x,nSamples,2);
  */
  //Copy x to host
  checkCudaErrors(cudaMemcpyAsync(d_x, h_x, mem_size_x, cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  
 
  //Generate range of x
  size_t xInitBlocks = (nSamples + block_size - 1) / block_size;
  xInitRange<<<xInitBlocks, block_size>>>(d_x,-100,100,nSamples);
  cudaDeviceSynchronize();
  

  //Allocate host Vandermonde matrix 
  dim3 dimsV(order+1,nSamples,1);
  unsigned int mem_size_V = sizeof(double) * dimsV.x * dimsV.y;
  double *h_V;
  checkCudaErrors(cudaMallocHost(&h_V, mem_size_V));
  
  
  //Allocate device Vandermonde matrix 
  double *d_V;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_V), mem_size_V));
  
  //Copy Vandermonde matrix to device
  checkCudaErrors(cudaMemcpyAsync(d_V, h_V, mem_size_V, cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  
  //Initialize Vandermonde matrix
  dim3 blocksVandermonde((dimsV.x + block_size - 1) / block_size, (dimsV.y + block_size - 1) / block_size, 1);
 Vandermonde<<<blocksVandermonde,threads>>>(d_x,d_V,order,nSamples);
	checkCudaErrors(cudaDeviceSynchronize());

    
      // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_x, d_x, mem_size_x, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpyAsync(h_V, d_V, mem_size_V, cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaDeviceSynchronize());
      
      for(int i=0;i<nSamples;i++){
      for(int j=0;j<order+1;j++){
  	printf("%.2f ",h_V[i*(order+1)+j]);
  }
  printf("\n");
  }
  
  checkCudaErrors(cudaFreeHost(h_V));
    checkCudaErrors(cudaFreeHost(h_x));
    checkCudaErrors(cudaFree(d_V));
  checkCudaErrors(cudaFree(d_x));
  
}

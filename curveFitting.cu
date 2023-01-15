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

//matrix operation using GPU
#include "matrixOperationsGPU.cuh"

#define BLOCK_SIZE 32

void inverseMatrixGPU(double *A, double *inv_A, int size);

template<typename T>
void printMatrix(T *A, int rows, int cols);

template<typename T>
void printDeviceMatrix(T* d_A, int rows, int cols);



int main(){
	/*printf("[Matrix Multiply Using CUDA] - Starting...\n");

  

  dim3 dimsA(5 * 2 * BLOCK_SIZE, 16, 1);
  dim3 dimsB(16, 5 * 2 * BLOCK_SIZE, 1);

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }



  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
         dimsB.x, dimsB.y);

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(BLOCK_SIZE, dimsA, dimsB);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);*/
  dim3 threads(BLOCK_SIZE,BLOCK_SIZE,1);
  //dim3 blocks((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  
  int nSamples=3;
  int order=6;
  
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  
  //Allocate host memory for x,y and B matrix
  double *h_x, *h_y, *h_B;
  
  unsigned int mem_size_samples = sizeof(double) * nSamples;
  checkCudaErrors(cudaMallocHost(&h_x, mem_size_samples));
  checkCudaErrors(cudaMallocHost(&h_y, mem_size_samples));
  
  unsigned int mem_size_B = sizeof(double) * (order + 1);
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  
  //Allocate device x
  double *d_x;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_x), mem_size_samples));
  
  /*
  //constant init x
  ConstantInit(h_x,nSamples,2);
  */
  
  //Copy x from Host to Device
  checkCudaErrors(cudaMemcpyAsync(d_x, h_x, mem_size_samples, cudaMemcpyHostToDevice, stream));
  
 
  //Generate range of x
  size_t xInitBlocks = (nSamples + BLOCK_SIZE - 1) / BLOCK_SIZE;
  xInitRange<<<xInitBlocks, BLOCK_SIZE, 0, stream>>>(d_x,-100,100,nSamples);
  checkCudaErrors(cudaStreamSynchronize(stream));

  //Allocate host Vandermonde matrix 
  dim3 dimsV(order+1,nSamples,1);
  unsigned int mem_size_V = sizeof(double) * dimsV.x * dimsV.y;
  double *h_V;
  checkCudaErrors(cudaMallocHost(&h_V, mem_size_V));
  
  
  //Allocate device Vandermonde matrix 
  double *d_V;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_V), mem_size_V));
  
  //Copy Vandermonde matrix to Device
  checkCudaErrors(cudaMemcpyAsync(d_V, h_V, mem_size_V, cudaMemcpyHostToDevice, stream));
  
  //Initialize Vandermonde matrix
  dim3 blocksVandermonde((dimsV.x + BLOCK_SIZE - 1) / BLOCK_SIZE, (dimsV.y + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  Vandermonde<<<blocksVandermonde, threads, 0, stream>>>(d_x, d_V, order, nSamples);
  checkCudaErrors(cudaStreamSynchronize(stream));
  
  //printDeviceMatrix(d_V,dimsV.y,dimsV.x);

    
      // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_x, d_x, mem_size_samples, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync(h_V, d_V, mem_size_V, cudaMemcpyDeviceToHost, stream));
    
    checkCudaErrors(cudaStreamSynchronize(stream));
      
     printDeviceMatrix(h_V,dimsV.y,dimsV.x);
  
  //test inverse matrix

  unsigned int mem_size = sizeof(double) * nSamples * nSamples;
  double *h_A, *h_C;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size));
  checkCudaErrors(cudaMallocHost(&h_C, mem_size));
  
  ConstantInit(h_A,nSamples*nSamples,2);
  h_A[2]=5.0;
  h_A[4]=4.0;
  h_A[5]=0.0;
  h_A[8]=1.0;
  
  for(int i=0;i<nSamples;i++){
      for(int j=0;j<nSamples;j++){
  	printf("%.2f ",h_A[i*nSamples+j]);
  }
  printf("\n");
  }
  
  double *d_A, *d_C;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size));
  
  checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_C, h_C, mem_size, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
  
  inverseMatrixGPU(d_A, d_C, nSamples);
  

  
  checkCudaErrors(cudaFreeHost(h_V));
    checkCudaErrors(cudaFreeHost(h_x));
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_V));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_C));
  
}

void inverseMatrixGPU(double *A, double *inv_A, int size){

	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  
	//Allocate identity and auxiliary matrix on device
	int mem_size = size * size * sizeof(double);
	double *d_I, *d_Aux, *d_ref, *d_ref2;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_I), mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_Aux), mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_ref), mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_ref2), mem_size));
	
	checkCudaErrors(cudaMemcpyAsync(d_ref, A, mem_size, cudaMemcpyDeviceToDevice, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
	
	//Initialize identity matrix
	dim3 threads(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 blocks((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
	
	initIdentityMatrix<<<blocks, threads, 0, stream>>>(d_I,size);
	checkCudaErrors(cudaStreamSynchronize(stream));
	
	for(int i=0;i<size;i++){
		reduceRow<<<blocks, threads, 0, stream>>>(d_ref,d_I,d_Aux,size,i);
		reduceRow<<<blocks, threads, 0, stream>>>(d_ref,d_ref,d_ref2,size,i);

		substractRow<<<blocks, threads, 0, stream>>>(d_ref2,d_Aux,d_I,size,i);
		substractRow<<<blocks, threads, 0, stream>>>(d_ref2,d_ref2,d_ref,size,i);
	}
	checkCudaErrors(cudaStreamSynchronize(stream));
	
	checkCudaErrors(cudaMemcpyAsync(inv_A, d_I, mem_size, cudaMemcpyDeviceToDevice, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
	
	//printf("inverse matrix\n");
	//printDeviceMatrix(inv_A,size,size);
  
	checkCudaErrors(cudaFree(d_I));
	checkCudaErrors(cudaFree(d_Aux));
	checkCudaErrors(cudaFree(d_ref));
	checkCudaErrors(cudaFree(d_ref2));
  
}

template<typename T>
void printMatrix(T *A, int rows, int cols){

	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			std::cout<<A[i*cols+j]<<" ";
		}
		
		printf("\n");
		
	}
}

template<typename T>
void printDeviceMatrix(T* d_A, int rows, int cols){
	
	T *temp;
	int mem_size = rows * cols * sizeof(T);
	checkCudaErrors(cudaMallocHost(&temp, mem_size));
	checkCudaErrors(cudaMemcpyAsync(temp, d_A, mem_size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	
	printMatrix(temp, rows, cols);
	
	checkCudaErrors(cudaFreeHost(temp));
}

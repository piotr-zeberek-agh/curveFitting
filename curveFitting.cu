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
	
  dim3 threads(BLOCK_SIZE,BLOCK_SIZE,1);
  
  int nSamples=11;
  int order=12;
  
  //Stream for synchronization and timing 
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  
  //Allocate Host x,y and B vectors
  double *h_x, *h_y, *h_B;
  
  unsigned int mem_size_samples = sizeof(double) * nSamples;
  checkCudaErrors(cudaMallocHost(&h_x, mem_size_samples));
  checkCudaErrors(cudaMallocHost(&h_y, mem_size_samples));
  
  unsigned int mem_size_B = sizeof(double) * (order + 1);
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  
  //Put some values into B and print
  for(int i=0; i<order+1; i++){
  	h_B[i]=i%3;
  }
  
  printf("B = ");
  printMatrix(h_B,1,order+1);
  
  //Allocate Device x,y and B vectors
  double *d_x, *d_y, *d_B;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_x), mem_size_samples));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_y), mem_size_samples));
  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
   
  //Copy x,y and B from Host to Device
  checkCudaErrors(cudaMemcpyAsync(d_x, h_x, mem_size_samples, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_y, h_y, mem_size_samples, cudaMemcpyHostToDevice, stream));
  
  checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));
  
  //Generate range of x and print
  double start = -5.0f;
  double stop = 5.0f;
  size_t xInitBlocks = (nSamples + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  xInitRange<<<xInitBlocks, BLOCK_SIZE, 0, stream>>>(d_x,start,stop,nSamples);
  checkCudaErrors(cudaStreamSynchronize(stream));
  
  printf("x = ");
  printDeviceMatrix(d_x,1,nSamples);

  //Allocate host Vandermonde matrix 
  dim3 dimsV(order+1,nSamples,1);
  
  double *h_V;
  unsigned int mem_size_V = sizeof(double) * dimsV.x * dimsV.y;
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
  
  //Calculate y=V*B and print
  dim3 blocksY((nSamples + BLOCK_SIZE - 1) / BLOCK_SIZE, (dimsV.y + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  matMul<<<blocksY,threads>>>(d_V, d_B, d_y, dimsV.y, dimsV.x, 1);
  checkCudaErrors(cudaStreamSynchronize(stream));
  
  printf("y = ");
  printDeviceMatrix(d_y,1,nSamples);
  
  checkCudaErrors(cudaFreeHost(h_x));
  checkCudaErrors(cudaFreeHost(h_y));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_V));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_V));
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
	checkCudaErrors(cudaDeviceSynchronize());
	
	printMatrix(temp, rows, cols);
	
	checkCudaErrors(cudaFreeHost(temp));
}

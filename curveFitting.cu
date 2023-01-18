// System includes
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <cmath>
#include <iomanip>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

//matrix operation using GPU
#include "matrixOperationsGPU.cuh"

#define BLOCK_SIZE 32

void invertMatrixGPU(double* A, int size);

void readData(double *x, double *y, const char *file_name, int nSamples);

void demo(int nSamples, int order);

void regression(double* x, double* y, int nSamples, int order, double* d_V=nullptr);

template<typename T>
void printMatrix(T *A, dim3 dims);

template<typename T>
void printDeviceMatrix(T* d_A, dim3 dims);

int main(int argc, char *argv[]){
	int nSamples=0;
	int order=0;
	size_t mem_size_samples = 0;
	const char *file_name = nullptr;
	
	
	switch(argc){
		case 1:	nSamples=64;
			order=2;
			printf("Running demo with %d samples and order of %d\n",nSamples,order);
		  	demo(nSamples,order);
		  	break;
		  	
		case 4:	//Read execution parameters
			file_name = argv[1];
			nSamples = atoi(argv[2]);
			order = atoi(argv[3]);
			printf("Polynomial regression for (x,y) pairs from \"%s\", number of samples: %d, order of polynomial: %d\n",file_name,nSamples,order);
			
			//Allocate Host x,y vector
			double *h_x, *h_y;
			mem_size_samples = nSamples * sizeof(double);
			checkCudaErrors(cudaMallocHost(&h_x, mem_size_samples));
			checkCudaErrors(cudaMallocHost(&h_y, mem_size_samples));
			
			//Read data from file
			readData(h_x, h_y, file_name, nSamples);

			//Perform regression
			regression(h_x, h_y, nSamples, order);
			break;
			
		default: printf("Wrong execution\n");
	}
}


void demo(int nSamples, int order){

  dim3 threads(BLOCK_SIZE,BLOCK_SIZE,1);
  
  //Stream for synchronization
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  
  //Allocate Host B vector
  double *h_B;
  dim3 dimsB(1,order+1,1);
  size_t mem_size_B = dimsB.x * dimsB.y * sizeof(double);
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  
  //Put some values into B
  for(int i=0; i<order+1; i++){
  	h_B[i]=i%3;
  }
  
  printf("B:\n");
  for(int i=0; i<dimsB.y; i++){
  	printf("%6.4f ",h_B[i]);
  }
  printf("\n");
  
  //Allocate Device x, y, and B vectors
  size_t mem_size_samples = sizeof(double) * nSamples;
  double *d_x, *d_y, *d_B;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_x), mem_size_samples));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_y), mem_size_samples));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));

  //Copy Host B into Device B
  checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

  //Generate range of x
  double start = -5.0f;
  double stop = 5.0f;
  size_t xInitBlocks = (nSamples + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  xInitRange<<<xInitBlocks, BLOCK_SIZE, 0, stream>>>(d_x,start,stop,nSamples);
  
  //Allocate device Vandermonde matrix
  double *d_V;
  dim3 dimsV(order+1,nSamples,1);
  size_t mem_size_V = sizeof(double) * dimsV.x * dimsV.y;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_V), mem_size_V));
  
  //Initialize Vandermonde matrix
  dim3 blocksVandermonde((dimsV.x + BLOCK_SIZE - 1) / BLOCK_SIZE, (dimsV.y + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  Vandermonde<<<blocksVandermonde, threads, 0, stream>>>(d_x, d_V, order, nSamples);
  
  //Calculate y=V*B
  dim3 blocksY(1, (dimsV.y + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  matMulGPU<<<blocksY,threads,0,stream>>>(d_V, d_B, d_y, dimsV.y, dimsV.x, 1);
  checkCudaErrors(cudaStreamSynchronize(stream));
      
  //Perform regression
  regression(d_x,d_y,nSamples,order,d_V);
  
  //Free memory
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFree(d_B));
}

void regression(double* x, double* y, int nSamples, int order, double* d_V){

	//Stream for synchronization
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));\
	
	double *d_x, *d_y;
	dim3 threads(BLOCK_SIZE,BLOCK_SIZE,1);
	
	dim3 dimsV(order+1,nSamples,1);
	size_t mem_size_V = sizeof(double) * dimsV.x * dimsV.y;
	dim3 blocksVandermonde((dimsV.x + BLOCK_SIZE - 1) / BLOCK_SIZE, (dimsV.y + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
	
	if(!d_V){
		//Allocate Device x, y
		size_t mem_size_samples = sizeof(double) * nSamples;
		
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_x), mem_size_samples));
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_y), mem_size_samples));
		
		//Copy data from Host to Device x,y
  		checkCudaErrors(cudaMemcpyAsync(d_x, x, mem_size_samples, cudaMemcpyHostToDevice, stream));
  		checkCudaErrors(cudaMemcpyAsync(d_y, y, mem_size_samples, cudaMemcpyHostToDevice, stream));
		
		//Allocate device Vandermonde matrix
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_V), mem_size_V));
		  
		//Initialize Vandermonde matrix
		Vandermonde<<<blocksVandermonde, threads, 0, stream>>>(d_x, d_V, order, nSamples);
		
	}else{
		d_x=x;
		d_y=y;
	}
	
  //Allocate device transposed Vandermonde matrix
  double *d_V_T;
  dim3 dimsV_T(nSamples,order+1,1);
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_V_T), mem_size_V));
  
  //Transpose Vandermonde matrix
  transposeGPU<<<blocksVandermonde,threads,0,stream>>>(d_V,d_V_T,dimsV.y,dimsV.x);
  
  //Allocate device matrix temp for storing temporary outcome
  //temp = V_T * V
  double *d_temp;
  size_t mem_size_temp = dimsV_T.y * dimsV.x * sizeof(double);
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_temp), mem_size_temp));
  
  //Calculate temp = V_T * V
  dim3 blocksTemp((dimsV.x + BLOCK_SIZE - 1) / BLOCK_SIZE, (dimsV_T.y + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  matMulGPU<<<blocksTemp,threads,0,stream>>>(d_V_T,d_V,d_temp,dimsV_T.y,dimsV_T.x,dimsV.x);
  
  //Synchronize before calling matrix inverting function
  checkCudaErrors(cudaStreamSynchronize(stream));
  
  //Invert matrix temp = V_T * V
  invertMatrixGPU(d_temp,dimsV.x);
  
  //Allocate device matrix temp2 for storing temporary outcome
  //temp2 = temp * V_T
  double *d_temp2;
  size_t mem_size_temp2 = dimsV.x * dimsV_T.x * sizeof(double);
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_temp2), mem_size_temp2));
  
  //Calculate temp2 = temp * V_T
  dim3 blocksTemp2((dimsV_T.x + BLOCK_SIZE - 1) / BLOCK_SIZE, (dimsV.x + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  matMulGPU<<<blocksTemp2,threads,0,stream>>>(d_temp,d_V_T,d_temp2,dimsV.x,dimsV.x,dimsV_T.x);
  
  //Allocate Device B_est vector
  double *d_B_est;
  dim3 dimsB(1,order+1,1);
  size_t mem_size_B = sizeof(double) * (order+1);
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B_est), mem_size_B));
  
  //Calculate B_est = temp2 * y
  dim3 blocksB(1, (dimsV.x + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  matMulGPU<<<blocksB,threads,0,stream>>>(d_temp2,d_y,d_B_est,dimsV.x,nSamples,1);
  
  //Allocate Host B_est vector
  double *h_B_est;
  checkCudaErrors(cudaMallocHost(&h_B_est, mem_size_B));

  //Copy data from Device to Host B_est
  checkCudaErrors(cudaMemcpyAsync(h_B_est, d_B_est, mem_size_B, cudaMemcpyDeviceToHost, stream));
  
  //Synchronize before retriving results calling matrix printing function
  checkCudaErrors(cudaStreamSynchronize(stream));

  //Print B_est
  printf("B_est:\n");
  for(int i=0; i<dimsB.y; i++){
  	printf("%6.4f ",h_B_est[i]);
  }
  printf("\n");
	
  //Free memory
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFreeHost(h_B_est));
  checkCudaErrors(cudaFree(d_B_est));
  checkCudaErrors(cudaFree(d_V));
  checkCudaErrors(cudaFree(d_V_T));
  checkCudaErrors(cudaFree(d_temp));
  checkCudaErrors(cudaFree(d_temp2));
}

void invertMatrixGPU(double *A, int size){

	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  
	//Allocate reference and temporary matrices on device
	size_t mem_size = size * size * sizeof(double);
	double *d_temp, *d_ref, *d_ref2;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_temp), mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_ref), mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_ref2), mem_size));
	
	//Copy data from matrix A to reference matrix
	checkCudaErrors(cudaMemcpyAsync(d_ref, A, mem_size, cudaMemcpyDeviceToDevice, stream));
	
	//Initialize Aux as identity matrix
	dim3 threads(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 blocks((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
	
	initIdentityMatrix<<<blocks, threads, 0, stream>>>(A,size);
	
	//Perform matrix inversion using Gauss-Jordan elimination
	for(int i=0;i<size;i++){
		reduceRow<<<blocks, threads, 0, stream>>>(d_ref,A,d_temp,size,i);
		reduceRow<<<blocks, threads, 0, stream>>>(d_ref,d_ref,d_ref2,size,i);

		substractRow<<<blocks, threads, 0, stream>>>(d_ref2,d_temp,A,size,i);
		substractRow<<<blocks, threads, 0, stream>>>(d_ref2,d_ref2,d_ref,size,i);
	}
	
	//Synchronize output
	checkCudaErrors(cudaStreamSynchronize(stream));
  
  	//Free memory
	checkCudaErrors(cudaFree(d_temp));
	checkCudaErrors(cudaFree(d_ref));
	checkCudaErrors(cudaFree(d_ref2));
}

void readData(double *x, double *y, const char *file_name, int nSamples){

	std::ifstream file(file_name);

	if ( file.is_open() ){

		for(int i=0; i<nSamples; i++){
			file >> x[i] >> y[i];
		}
	}else{
		printf("Cannot read file %s",file_name);
	}	

}

template<typename T>
void printMatrix(T *A, dim3 dims){

	for(int i=0;i<dims.y;i++){
		for(int j=0;j<dims.x;j++){
			std::cout<<std::setprecision(20)<<A[i*dims.x+j]<<" ";
		}
		
		printf("\n");
	}
}

template<typename T>
void printDeviceMatrix(T* d_A, dim3 dims){
	
	T *temp;
	size_t mem_size = dims.x * dims.y * sizeof(T);
	
	checkCudaErrors(cudaMallocHost(&temp, mem_size));
	checkCudaErrors(cudaMemcpyAsync(temp, d_A, mem_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());
	
	printMatrix(temp, dims);
	
	checkCudaErrors(cudaFreeHost(temp));
}

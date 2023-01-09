#include <iostream>

__global__ void hello(){
    printf("hello\n");
}

int main(){
	printf("Say \"Hello\" 4 times\n");
	hello<<<2,2>>>();
	cudaDeviceSynchronize();
}

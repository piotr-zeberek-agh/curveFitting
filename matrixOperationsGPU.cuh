#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

__global__ void xInitRange(double *x, double start, double end, int size)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double step = (end - start) / (size - 1);

    if (idx < size)
    {
        x[idx] = start + idx * step;
    }
}

__global__ void vandermonde(double *x, double *V, int degree, int nSamples)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < degree + 1 && row < nSamples)
    {
        int pos = row * (degree + 1) + col;
        V[pos] = pow(x[row], col % (degree + 1));
    }
}

__global__ void transpose(double *A, double *A_T, int rows, int cols)
{
	//Sub matrix for A
	__shared__ double block[BLOCK_SIZE][BLOCK_SIZE+1];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	
	//Load matrix A to shared memory
	int xIndex = bx * BLOCK_SIZE + tx;
	int yIndex = by * BLOCK_SIZE + ty;
	
	if(tx < cols && ty < rows)
	{
		int idx = yIndex * cols + xIndex;
		block[ty][tx] = A[idx];
	}

        // synchronise to ensure all data is written
	__syncthreads();

	// Write the data from tiles to transposed ouput matrix A_T
	xIndex = by * BLOCK_SIZE + tx;
	yIndex = bx * BLOCK_SIZE + ty;
	
	if(xIndex < rows && yIndex < cols)
	{
		int idx_T = yIndex * rows + xIndex;
		A_T[idx_T] = block[tx][ty];
	}
}

__global__ void matMul(double * A, double * B, double * C, int m, int k, int n)
{
    //Sub matrices for A and B
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //Variable for storing partial sum for C 
    double Csub = 0;

    for (int i = 0; i < (k-1)/BLOCK_SIZE +1; ++i)
    {
        //Load matrix A to shared memory
        if(row < m && i*BLOCK_SIZE+tx < k)
            As[ty][tx] = A[row*k + i*BLOCK_SIZE+tx];
        else
            As[ty][tx] = 0.0f;

        //Load matrix B to shared memory
        if (i*BLOCK_SIZE+ty < k && col < n)
            Bs[ty][tx] = B[(i*BLOCK_SIZE+ty)*n + col];
        else
            Bs[ty][tx] = 0.0f;

	//Synchronize threads before calculating sum
        __syncthreads();

        //Calculate partial sum
        for (int j = 0; j < BLOCK_SIZE; ++j)
            Csub += As[ty][j] * Bs[j][tx];
	
	//Synchronize to unsure calculations are finished
        __syncthreads();

    }
    
    //Copy calculated sum to C matrix
    if (row < m && col < n)
        C[row*n+col] = Csub;

}

__global__ void initIdentityMatrix(double *I, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < size && idy < size)
    {
        I[idy * size + idx] = (idx == idy ? 1 : 0);
    }
}

__global__ void reduceRow(double *ref, double *input, double *output, int size, int current)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < size && idy < size)
    {
        int pos = idy * size + idx;

        if (idy == current)
        {
            double currentValue = ref[current * size + current];

            if (currentValue)
            {
                output[pos] = input[pos] / currentValue;
            }
            else
            {
                printf("Math Error");
            }
        }
        else
        {
            output[pos] = input[pos];
        }
    }
}

__global__ void substractRow(double *ref, double *input, double *output, int size, int current)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < size && idy < size)
    {
        int pos = idy * size + idx;

        if (idy != current)
        {
            double factor = ref[idy * size + current];
            output[pos] = input[pos] - input[current * size + idx] * factor;
        }
        else
        {
            output[pos] = input[pos];
        }
    }
}

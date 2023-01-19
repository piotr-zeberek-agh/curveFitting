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

__global__ void vandermonde(double *x, double *V, int order, int nSamples)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < order + 1 && row < nSamples)
    {
        int pos = row * (order + 1) + col;
        V[pos] = pow(x[row], col % (order + 1));
    }
}

__global__ void transpose(double *A, double *A_T, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        int pos = idy * cols + idx;
        int pos_T = idx * rows + idy;

        A_T[pos_T] = A[pos];
    }
}

/*
__global__ void matMul(double *A, double *B, double *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0;

    if (col < n && row < m)
    {
        for (int i = 0; i < k; i++)
            sum += A[row * k + i] * B[i * n + col];

        C[row * n + col] = sum;
    }
}

*/
__global__ void matMul(double * A, double * B, double * C, int m, int k, int n)
{
    //Sub matrices for A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

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

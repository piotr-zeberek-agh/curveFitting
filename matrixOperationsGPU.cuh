

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

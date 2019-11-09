#include <stdio.h>
#include "realtime.h"

#ifdef __cplusplus
extern "C" {
#endif


#ifdef SET_DOUBLE
typedef double FLOAT_TYPE;
#else
typedef float FLOAT_TYPE;
#endif
#define BLOCK_SIZE   32
#define GET_MATRIX(m, row, col) (m.elements[row*m.cols + col])
// Versions of matrix computations
enum EMultVersions {SIMPLE, SHARED};

typedef struct{
    int cols;
    int rows;
    FLOAT_TYPE *elements;
} Matrix;

__device__
FLOAT_TYPE getElement(Matrix m, int row, int col)
{
    if (row < m.rows && col < m.cols)
    {
        return m.elements[m.cols*row + col];
    }
    else
    {
        return 0;
    }
}

__device__
void setElement(Matrix m, int row, int col, FLOAT_TYPE val)
{
    if (row < m.rows && col < m.cols)
    {
        m.elements[m.cols*row + col] = val;
    }
}

__global__
void sharedMult(Matrix a, Matrix b, Matrix res)
{
    // block ids
    int brow = blockIdx.y;
    int bcol = blockIdx.x;
    // id of thread in his block
    int trow = threadIdx.y;
    int tcol = threadIdx.x;

    FLOAT_TYPE sum = 0;
    for (int m = 0; m < a.cols; m += BLOCK_SIZE)
    {
        __shared__ FLOAT_TYPE aTile[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ FLOAT_TYPE bTile[BLOCK_SIZE][BLOCK_SIZE];

        aTile[trow][tcol] = getElement(a, brow*BLOCK_SIZE + trow, tcol + m); 
        bTile[trow][tcol] = getElement(b, trow + m, bcol*BLOCK_SIZE + tcol);
        __syncthreads();

        for(int k = 0;  k < BLOCK_SIZE; ++k)
        {
            sum += aTile[trow][k] * bTile[k][tcol];
        }
        __syncthreads();
    }
    setElement(res, 
               blockIdx.y * blockDim.y + threadIdx.y,
               blockIdx.x * blockDim.x + threadIdx.x,
               sum);
}

__global__
void simpleMult(Matrix a, Matrix b, Matrix res)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= a.rows || col >= b.cols)
    {
        return;
    }

    FLOAT_TYPE sum = 0;
    for (int k = 0; k < a.cols; ++k)
    {
        sum += getElement(a, row, k) * getElement(b,  k, col);
    }
    setElement(res, row, col, sum);
}

/// <summary>checkes the last cuda command. if it wasn't successful
/// stops the program with error description</summary>
/// <param name="line">optional paramer with line number(where 
/// the function was called</param>
inline void checkErr(int line = -1)
{
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) 
    {
        if (line != -1)
            fprintf(stderr, "line nr. %d:", line );
        fprintf(stderr, "Cuda error: %s.\n", cudaGetErrorString( err) );
        exit(-1);
    }                         
}

/// <summary>computes depending on the size of the problem matrix </summary>
/// <returns>min number of blocks necessary to solve the problem</returns>
dim3 numberOfBlocks(dim3 threadsPerBlock, int rows, int cols)
{
    dim3 numberOfBlocks(1,1,1);
    
    if(cols%threadsPerBlock.x)
    {
        numberOfBlocks.x = cols/threadsPerBlock.x + 1;
    }
    else
    {
        numberOfBlocks.x = cols/threadsPerBlock.x;
    }

    if(rows%threadsPerBlock.y)
    {
        numberOfBlocks.y = rows/threadsPerBlock.y + 1;
    }
    else
    {
        numberOfBlocks.y = rows/threadsPerBlock.y;
    }
    return numberOfBlocks;
}

/// <summary>computes a * b on GPU and stores the result in res</summary>
/// <param name="version">version of kernel </param>
/// <param name="threadsPerBlock">size of block. Global size would be adjust 
/// automatically</param>
/// <param name="a">input Matrix a</param>
/// <param name="b">input Matrix b</param>
/// <param name="res">output Matrix</param>
void matirxMultOnGPU(EMultVersions version,         // kernel version
                     const dim3 &threadsPerBlock,   // blocksize
                     const Matrix &a,              // Matrix A
                     const Matrix &b,              // Matrix B
                     Matrix &res)                  // Matrix res = A*B
{
    // compute the size of the result matrix
    res.rows = a.rows;
    res.cols = b.cols;
    res.elements = (FLOAT_TYPE*)malloc(res.cols*res.rows*sizeof(FLOAT_TYPE));
    // allocate memory on GPU for matrices A, B and Res
    size_t sizeOfA = a.rows*a.cols * sizeof(FLOAT_TYPE);
    Matrix deviceA;
    deviceA.cols = a.cols;
    deviceA.rows = a.rows;
    cudaMalloc((void**)&deviceA.elements, sizeOfA);
    checkErr(__LINE__);

    size_t sizeOfB = b.rows*b.cols * sizeof(FLOAT_TYPE);
    Matrix deviceB;
    deviceB.cols = b.cols;
    deviceB.rows = b.rows;
    cudaMalloc((void**)&deviceB.elements, sizeOfB);
    checkErr(__LINE__);
    size_t sizeOfRes = res.rows*res.cols * sizeof(FLOAT_TYPE);
    Matrix deviceRes;
    deviceRes.cols = res.cols;
    deviceRes.rows = res.rows;
    cudaMalloc((void**)&deviceRes.elements, sizeOfRes);
    checkErr(__LINE__);
    // copy data on GPU
    cudaMemcpy(deviceA.elements, a.elements, sizeOfA,
        cudaMemcpyHostToDevice);
    checkErr(__LINE__);
    cudaMemcpy(deviceB.elements, b.elements, sizeOfB, 
        cudaMemcpyHostToDevice);
    checkErr(__LINE__);
    dim3 blocksPerGrid =
        numberOfBlocks(threadsPerBlock, res.rows, res.cols);
    // data size in GB witch has to be written/read on/from GPU
    double dataSize = (a.rows*a.cols+b.cols*b.rows + a.rows*b.cols) * 
        sizeof(FLOAT_TYPE)/(1e9);
    double numberOfOperations = b.cols*a.rows*(2*a.cols-1.0)/(1e9);

    printf("Input information\n");
    printf("Matrix A: %dx%d\n",a.cols, a.cols);
    printf("Matrix B: %dx%d\n",b.rows, b.cols);
    printf("Memory transport(GB):%f\n",dataSize);
    printf("Number of operations(*1e-9): %f\n\n", numberOfOperations);

    double s, e;
    s = GetRealTime();
    // size of sharedMemory array from shredMult kernel
    switch(version)
    {
    case SIMPLE:
        simpleMult<<<blocksPerGrid, threadsPerBlock>>>
            (deviceA, deviceB, deviceRes);
        break;
    case SHARED:
        sharedMult<<<blocksPerGrid, threadsPerBlock>>>
            (deviceA, deviceB, deviceRes);
        break;
    }
    // makes sure, that computation is already ready
    cudaThreadSynchronize();
    e = GetRealTime();
    printf("time on GPU (without memory transport and init):%f\n", e - s);
    printf("GFlops: %f\n\n", numberOfOperations / (e-s));
    //copy computing result on host
    cudaMemcpy(res.elements, deviceRes.elements, sizeOfRes, 
        cudaMemcpyDeviceToHost);
    checkErr(__LINE__);

    cudaFree(deviceA.elements);
    cudaFree(deviceB.elements);
    cudaFree(deviceRes.elements);
}

///<summary>version consists of text description of EMultVersion</summary>
///<return>SHARED or SIMPLE. If strVersion contains wrong description, then
/// SIMPLE</return>
EMultVersions getAlgoVersion(const char* strVersion)
{
    const char *shared = "SHARED";
    EMultVersions version = SIMPLE;
    if (strcmp(shared, strVersion) == 0)
        version = SHARED;
    return version;
}

/// <summary>reserves memory for m.cols*m.rows elements of Type FLOAT_TYPE
/// fills m with random values from [0,..,1]</summary>
/// <param name="m">pointer to matrix which should be inialized</param>
/// <param name="rows">number of rows in m</param>
/// <param name="cols">number of columns in m</param>
void fill(Matrix* m, int rows, int cols)
{
    m->cols = cols;
    m->rows = rows;

    size_t numberOfElements = cols * rows;
    m->elements = (FLOAT_TYPE*)malloc(numberOfElements * sizeof(FLOAT_TYPE));
    for (size_t i = 0; i < numberOfElements; ++i)
    {
        m->elements[i] = ((FLOAT_TYPE) (rand() % 101))/100;
    }
}

/// <summary>frees memory referenced by m.elements</summary>
/// <param name="m">Matrix with allocated memory</param>
void freeMatrix(Matrix m)
{
    free(m.elements);
}

/// <summary>compares res with a*b. prints first fail</summaru>
/// <param name="a">matrix a</param>
/// <param name="b">matrix b</param> 
/// <param name="res">the matrix, which should be controled</param>
bool compare(Matrix a, Matrix b, Matrix res)
{
    if (res.cols != b.cols || res.rows != a.rows)
        return false;
    for (int row = 0; row < a.rows; ++ row)
    {
        for (int col = 0; col < b.cols; ++ col)
        {
            FLOAT_TYPE sum = 0;
            for (int k = 0; k < a.cols; ++k)
            {
                sum += GET_MATRIX(a, row, k) * GET_MATRIX(b, k, col);
            }
            if (fabs(sum-GET_MATRIX(res, row, col))/fabs(GET_MATRIX(res, row, col)) > 0.0001)
            {
                printf("from CPU: %.10f, from GPU %.10f\n", 
                       sum, 
                       GET_MATRIX(res, row, col));
                return false;
            }
        }
    }
    return true;
}

/// <summary>usage executable MATRIXSIZE VERSION RESALTTEST</summary>
int matmul(int deviceId, size_t size)
{
    // dummy function in order to create context
    cudaFree(0);

    EMultVersions version = SHARED;
    Matrix a;
    Matrix b;
    int testResult = 0;
    printf("start matrix mult computation\n");
#ifdef SET_DOUBLE
    printf("set double(FloatType = double) precision\n");
#else
    printf("set single(FloatType = float) precision with CUDA\n");
#endif
    // create test data
//    if (argc == 1) // default behavior
//    {
        fill(&a, size, size);
        fill(&b, size, size);
//    }    
//    else if (argc == 2) // choose dimension of the matrix
//    {
//        int matrixDim = atoi(argv[1]);
//        fill(&a, matrixDim, matrixDim);
//        fill(&b, matrixDim, matrixDim);
//    }
//    else if (argc == 3) // choose algorithm and dimension of the matrix
//    {
//        int matrixDim = atoi(argv[1]);
//        fill(&a, matrixDim, matrixDim);
//        fill(&b, matrixDim, matrixDim);
//        version = getAlgoVersion(argv[2]);
//    }
//    else // choose algorithm and dimension of the matrix, and test behavior
//    {
//        int matrixDim = atoi(argv[1]);
//        fill(&a, matrixDim, matrixDim);
//        fill(&b, matrixDim, matrixDim);
//        version = getAlgoVersion(argv[2]);
//        testResult = atoi(argv[3]);
//    }
    cudaSetDevice(deviceId);
    Matrix res = {0,0,0};
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    // variables for time measurement
    double s, e;
    s = GetRealTime();
    switch (version)
    {
    case SIMPLE:
        printf("simple version.\n");
        matirxMultOnGPU(SIMPLE, threadsPerBlock, a, b, res);
        break;
    case SHARED:
        printf("shared version.\n");
        matirxMultOnGPU(SHARED, threadsPerBlock, a, b, res);
        break;
    }
    e = GetRealTime();  
    double numberOfOperations = b.cols*a.rows*(2*a.cols - 1.0)/(1e9);
    printf("total time (GPU init, memory reservation and computation): %f\n", e - s);
    printf("GFlops(GPU init, memory reservation and computation)): %f\n\n", numberOfOperations / (e-s));
    if (testResult)
    {
        printf("start test: ");
        printf((compare(a, b, res)?"passed\n":"fail\n"));
    }
    freeMatrix(a);
    freeMatrix(b);
    freeMatrix(res);
    printf("end matrix mult computation\n\n");
    return 0;
}

#ifdef __cplusplus
}
#endif

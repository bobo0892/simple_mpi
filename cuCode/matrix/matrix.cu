#include "cublas_v2.h"

#include "matrix.hpp"
#include "cuda_helper.hpp"
#include "CUDATimer.h"

#define H2D(dst, src, dataSize) checkCUDAError(cudaMemcpy(dst, src, dataSize, cudaMemcpyHostToDevice))
#define D2H(dst, src, dataSize) checkCUDAError(cudaMemcpy(dst, src, dataSize, cudaMemcpyDeviceToHost))

static void initMatrix(const int rows, const int cols, float *matrix){
	for(size_t i = 0; i < rows*cols; ++i){
		matrix[i] = rand() / (float)RAND_MAX;
	}
}

void matrixSingle(int M, int K, size_t N, int device){

	float *matrixA = new float[M*K];
	float *matrixB = new float[K*N];
	float *matrixC = new float[M*N];

	initMatrix(M, K, matrixA);
	initMatrix(K, N, matrixB);
	initMatrix(M, N, matrixC);

	float *d_matrixA = NULL;
	float *d_matrixB = NULL;
	float *d_matrixC = NULL;
	checkCUDAError(cudaSetDevice(device));
	checkCUDAError(cudaMalloc((void**)&d_matrixA, sizeof(float)*M*K));
	checkCUDAError(cudaMalloc((void**)&d_matrixB, sizeof(float)*K*N));
	checkCUDAError(cudaMalloc((void**)&d_matrixC, sizeof(float)*M*N));
	H2D(d_matrixA, matrixA, sizeof(float)*M*K);
	H2D(d_matrixB, matrixB, sizeof(float)*K*N);
	H2D(d_matrixC, matrixC, sizeof(float)*M*N);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasOperation_t transa = CUBLAS_OP_N;
	cublasOperation_t transb = CUBLAS_OP_N;

	const float alpha = 0.5f;
	const float beta  = 1.0f;
	CUDATimer timer;
	timer.start();
	cublasStatus_t status = cublasSgemm(handle, transa, transb, M, N, K, &alpha, d_matrixA,
			M, d_matrixB, K, &beta, d_matrixC, M);
	timer.stop();
	float gflops = 2.0f*M*N*K / ((float)1000*1000) / timer.getElapsedMilliSeconds();
	std::cout << "CUDA Device " << device << " use time: " <<
		timer.getElapsedMilliSeconds() << "ms " << "Gflops: " << gflops << "GB/s" << std::endl;

	cublasDestroy(handle);

	checkCUDAError(cudaFree(d_matrixA));
	checkCUDAError(cudaFree(d_matrixB));
	checkCUDAError(cudaFree(d_matrixC));
}


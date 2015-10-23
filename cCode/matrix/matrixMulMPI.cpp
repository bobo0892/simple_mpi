#include<math.h>
#include<stdlib.h>
#include <assert.h>

#include<mpi.h>
#include<iostream>
#include <sys/time.h>

static double getMS() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec*1000.0 + tv.tv_usec/1000.0;
}

static void initMatrix(float *matrix, int rows, int cols) {
	assert(matrix);
	for(int i = 0; i < rows*cols; ++i){
		matrix[i] = rand()/(float)RAND_MAX;
	}
}

static void matrixMultiklySingle(const float *matrixA, 
		const float *matrixB, float *result, int m, int k, int n) {

    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            float temp = 0;
            for(int z=0; z<k; z++) {
                temp += matrixA[i*k+z] * matrixB[k*n + j];
            }
            result[i*n+j] = temp;
        }
    }
}


int main(int argc, char** argv)
{
	if(argc < 4){
		std::cout << "Usage:" << std::endl;
		std::cout << "exe m, k, n" << " or ";
		std::cout << "mkirun -np np exe m k n" << std::endl;
		exit(0);
	}
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    
    float *A, *B, *C;
    float *bA, *bC;  

    int myrank, numkrocs;

    MPI_Status status;
  
    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &numkrocs); 
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
    int bm = m / numkrocs;

    bA = new float[bm * k];
    B  = new float[k * n];
    bC = new float[bm * n];

	double st, et;
    if(myrank == 0){
        A = new float[m * k];
        C = new float[m * n];
        
        initMatrix(A, m, k);
        initMatrix(B, k, n);
		st = getMS();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatter(A, bm * k, MPI_FLOAT, bA, bm *k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    matrixMultiklySingle(bA, B, bC, bm, k, n);
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Gather(bC, bm * n, MPI_FLOAT, C, bm * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
    int remainRowsStartId = bm * numkrocs;
    if(myrank == 0 && remainRowsStartId < m){
        int remainRows = m - remainRowsStartId;
        matrixMultiklySingle(A + remainRowsStartId * k, B, C + remainRowsStartId * n, remainRows, k, n);
    }
  
    delete[] bA;
    delete[] B;
    delete[] bC;
    
    if(myrank == 0){
		et = getMS();
		std::cout << "matrix multiply use time " << et-st << "ms" << std::endl;
        delete[] A;
        delete[] C;
    }
    
    MPI_Finalize();  

    return 0;
}


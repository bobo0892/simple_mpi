#include <mpi.h>
#include "matrix.hpp"

int main(int argc, char* argv[]){

	int myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	if(0 == myrank){ 
		matrixSingle(2048, 2048, 2048, myrank);
	}
	if(1 == myrank){
		matrixSingle(2048, 2048, 2048, myrank);
	}
	if(2 == myrank){
		matrixSingle(2048, 2048, 2048, myrank);
	}

	MPI_Finalize();

	return 0;
}

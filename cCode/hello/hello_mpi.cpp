#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]){
	
	char *message = new char[20];
	int myrank;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	if(0 == myrank){
		strcpy(message, "Hello, mpi");
		MPI_Send(message, strlen(message)+1, MPI_CHAR, 1, 99, MPI_COMM_WORLD);
	}

	if(1 == myrank){
		MPI_Recv(message, 20, MPI_CHAR, 0, 99, MPI_COMM_WORLD, &status);
		std::cout << "received: " << message << std::endl;
	}

	MPI_Finalize();

	return 0;

}

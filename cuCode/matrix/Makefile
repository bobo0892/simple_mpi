CC := mpic++
FLAGS := -I/usr/local/openmpi/include -L/usr/local/openmpi/lib

NVCC := nvcc
NVFLAGS := -I/usr/local/cuda-7.0/include -L/usr/local/cuda-7.0/lib64 
LIBS := -lmpi -lpthread -lcudart -lcublas

test_matrix_mpi: matrix.o test_matrix_mpi.o
	$(CC) $(NVFLAGS) -o $@ $^ $(LIBS)

matrix.o : matrix.cu
	$(NVCC) $(NVFLAGS) -c $<

test_matrix_mpi.o:test_matrix_mpi.cpp
	$(CC) $(FLAGS) -c $<
clean:
	rm -rf *.o test_mpi_gpu 


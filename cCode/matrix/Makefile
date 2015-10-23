CC=mpicxx
CFLAGAS := -std=c++11
LDFLAGS := -lpthread

OBJECT := matrixMulMPI.o
TARGET := test_matrix_mul_mpi

$(TARGET) : $(OBJECT)
	$(CC) $(CFLAGAS) $^ -o $@ $(LDFLAGS)

%.o:%.cpp
	$(CC) $(CFLAGAS) -c $<

clean:
	rm -rf *.o test_matrix_mul_mpi

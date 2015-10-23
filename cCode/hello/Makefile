CC := mpic++
CFLAGS := -pthread

test_hello_mpi : hello_mpi.cpp
	$(CC) $(CFLAGS) -o $@ $^ 

clean:
	rm -rf *.o test_hello_mpi 

all: restart omp cuda mpi clean run

SRC_DIR    = ./src/
BIN_DIR    = ./bin/
OBJ_DIR    = ./obj/

#Generate OMP program binary
omp: omp.o
	g++ -Wall -O3 -fopenmp -o $(BIN_DIR)omp main_omp.o

omp.o: $(SRC_DIR)main_omp.cpp
	g++ -Wall -O3 -fopenmp -c $(SRC_DIR)main_omp.cpp

#Generate CUDA program binary
cuda: cuda.o
	nvcc -Wall -o $(BIN_DIR)cuda main_cuda.o

cuda.o: $(SRC_DIR)main_cuda.cpp
	nvcc -Wall -c $(SRC_DIR)main_cuda.cu

#Generate OpenMPI program binary
mpi: mpi.o
	g++ -Wall -o $(BIN_DIR)mpi main_mpi.o

mpi.o: $(SRC_DIR)main_mpi.cpp
	g++ -Wall -c $(SRC_DIR)main_mpi.cpp

#Move *.o to ./obj
clean:
	mv *o $(OBJ_DIR)

#Delete all *o and binaries.
restart:
	rm -f -r $(OBJ_DIR)*o
	rm -f $(BIN_DIR)*

run: 
	chmod 777 run.sh
	./run.sh
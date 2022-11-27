all: omp cuda mpi

SRC_DIR    = ./src/
BIN_DIR    = ./bin/
OBJ_DIR    = ./obj/

#Generate OMP program binary
omp: $(OBJ_DIR)omp.o
	gcc -Wall -o $(BIN_DIR)omp $(OBJ_DIR)omp.o

omp.o: $(SRC_DIR)main_omp.cpp
	gcc -Wall -c $(SRC_DIR)main_omp.cpp

#Generate CUDA program binary
cuda: $(OBJ_DIR)cuda.o
	gcc -Wall -o $(BIN_DIR)cuda $(OBJ_DIR)cuda.o

cuda.o: $(SRC_DIR)main_cuda.cpp
	gcc -Wall -c $(SRC_DIR)main_cuda.cpp

#Generate OpenMPI program binary
mpi: $(OBJ_DIR)mpi.o
	gcc -Wall -o $(BIN_DIR)mpi $(OBJ_DIR)mpi.o

mpi.o: $(SRC_DIR)main_mpi.cpp
	gcc -Wall -c $(SRC_DIR)main_mpi.cpp
#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>
#include <math.h>

using namespace std;

__global__ void multiply_matrices(float *A, float *B, float *C, int n){
    int COL = blockIdx.x*blockDim.x+threadIdx.x;
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;

    if (COL < n && ROW < n){
        float sum = 0;
        for (int k = 0; k < n; k++){
            sum += A[ROW*n + k] * B[k*n + COL];
        }
        C[ROW*n + COL] = sum;
    }
}

void print_matrix(float *matrix, int size, char name){
    cout << name << ":" << endl;
    for (int i = 0; i < size; i++){
        cout << "[ ";
        for (int j = 0; j < size; j++){
            cout << matrix[i*size + j] << " ";
        }
        cout << "]" << endl;
    }
    cout << endl;
}

int main(int argc, char *argv[]) {
    //Receives m, n, p. Matrices dimensions.
    int n = stoi(argv[1]);
    int THREADS = stoi(argv[2]);
    int size = n*n*sizeof(float);

    srand(time(0));
    
    float *A = (float *) malloc(size);
    float *B = (float *) malloc(size);
    float *C = (float *) malloc(size);

    //Initialize matrices
    //A[i][j] = A[i*ncols + j]
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i*n + j] = (float) rand()/RAND_MAX;
            B[i*n + j] = (float) rand()/RAND_MAX;
            C[i*n + j] = 0;
        }
    }

    //CUDA WORK
    //n*n threads distributed in n/THREADS blocks, each block has THREADS*THREADS threads.
    float *d_A , *d_B, *d_C;
    dim3 threads_per_block(THREADS, THREADS);
    dim3 num_blocks(ceil((double) n / threads_per_block.x), ceil((double) n / threads_per_block.y));
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    multiply_matrices<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    print_matrix(A, n, 'A');
    print_matrix(B, n, 'B');
    print_matrix(C, n, 'C');
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    
    return 0;
}

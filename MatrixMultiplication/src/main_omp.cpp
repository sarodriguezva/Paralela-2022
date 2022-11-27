#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>

using namespace std;

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
    srand(time(0));
    
    float *A = (float *) malloc(n*n*sizeof(float));
    float *B = (float *) malloc(n*n*sizeof(float));
    float *C = (float *) malloc(n*n*sizeof(float));
    
    //Initialize matrices
    //A[i][j] = A[i*ncols + j]
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i*n + j] = (float) rand()/RAND_MAX;
            B[i*n + j] = (float) rand()/RAND_MAX;
            C[i*n + j] = 0;
        }
    }
    
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            for (int k = 0; k < n; k++){
                C[i*n + j] += A[i*n+k] * B[k*n + j];
            }
        }
    }

    print_matrix(A, n, 'A');
    print_matrix(B, n, 'B');
    print_matrix(C, n, 'C');
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}

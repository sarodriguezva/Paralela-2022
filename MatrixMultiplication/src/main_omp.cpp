#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    //Receives m, n, p. Matrices dimensions.
    int n = stoi(argv[1]);
    srand(time(0));
    
    float *A = (float *) malloc(n*n*sizeof(float));
    float *B = (float *) malloc(n*n*sizeof(float));
    float *C = (float *) malloc(n*n*sizeof(float));
    
    //Initialize matrices
    for (int i = 0; i < n*n; i++){
        A[i] = (float) rand()/RAND_MAX;
        B[i] = (float) rand()/RAND_MAX;
        C[i] = 0;
    }
    
    //A[i][j] = A[i*ncols + j]
    
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            for (int k = 0; k < n; k++){
                C[i*n + j] += A[i*n+k] + B[k*n + j];
            }
        }
    }

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            cout << C[i*n + j] << endl;
        }
    }
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}

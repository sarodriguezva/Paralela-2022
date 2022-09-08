//gcc -Wall -O3 -fopenmp calcPi_OMP.c -o calcPi_OMP
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

#define THREADS 1
#define ITERATIONS 2e09

double pi = 0.0;

int main(){
    int i, j, N = ITERATIONS;
    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for reduction (+:pi)
        for (i = 0; i < N; i+=2){
            pi += (double)(4.0 / ((i*2)+1));
        }
        #pragma omp for reduction (+:pi) nowait
        for (j = 1; j < N-1; j+=2){
            pi -= (double)(4.0 / ((j*2)+1));
        }
    }

    printf("\npi: %2.12f \n", pi);
    
    return 0;
}
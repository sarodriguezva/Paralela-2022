#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>

#define THREADS 16
#define ITERATIONS 2e09

double piFrac[THREADS];
double pi;

void * calculate(void *arg){
    int init, end, rank, iter, threadId = *(int *)arg;

    rank = (ITERATIONS/THREADS);
    init = rank*threadId;
    end = init + rank - 1;

    piFrac[threadId] = 0.0;

    iter = init;
    while (iter < end){
        piFrac[threadId] += (double)(4.0 / ((iter*2)+1));
        iter++;
        piFrac[threadId] -= (double)(4.0 / ((iter*2)+1));
        iter++;
    }

    return 0;
}


int main(){
    int i;
    int threadId[THREADS];
    int *retval;
    pthread_t thread[THREADS];

    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);

    for (i = 0; i < THREADS; i++){
        threadId[i] = i;
        pthread_create(&thread[i], NULL, (void *) calculate, &threadId[i]);
    }

    for(i = 0; i < THREADS; i++){
        pthread_join(thread[i], (void **)&retval);
    }

    for(i = 0; i < THREADS; i++){
        pi += piFrac[i];
    }

    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    printf("\npi: %2.12f \n", pi);
    return 0;
}
#include <omp.h>
#include <iostream>

using namespace std;

int THREADS = 1;

int main(int argc, char *argv[]){
    THREADS = stoi(argv[1]);

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for
        for (int i = 0; i < 20; i++){
            int id = omp_get_thread_num();
            cout << id << endl;
        }
    }

    return 0;
}
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

#define N 100000

// Function to swap two elements in an array
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);

    return i + 1;
}

void quicksort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int arr[N];
    srand(time(NULL) + rank);  // Seed with different values for each process

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            arr[i] = rand() % 1000;
        }
    }

    MPI_Bcast(arr, N, MPI_INT, 0, MPI_COMM_WORLD);

    int chunksize = N / size;
    int buffer[chunksize];
    MPI_Scatter(arr, chunksize, MPI_INT, buffer, chunksize, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();  // Start timing
    quicksort(buffer, 0, chunksize - 1);
    double end_time = MPI_Wtime();    // End timing

    MPI_Gather(buffer, chunksize, MPI_INT, arr, chunksize, MPI_INT, 0, MPI_COMM_WORLD);

    double local_sort_time = end_time - start_time;
    double max_sort_time;
    MPI_Reduce(&local_sort_time, &max_sort_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Display only the execution time
        std::cout << "Execution Time: " << max_sort_time << " seconds" << std::endl;
    }

    MPI_Finalize();

    return 0;
}

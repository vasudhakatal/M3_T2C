#include <iostream>
#include <vector>
#include <ctime>
#include <mpi.h>

using namespace std;

// Function to partition the array
int partition(vector<int> &arr, int low, int high) {
    int pivot = arr[high];  // Choosing the last element as pivot
    int i = low - 1;  // Index of smaller element

    for (int j = low; j <= high - 1; j++) {
        // If current element is smaller than or equal to pivot
        if (arr[j] <= pivot) {
            i++;  // Increment index of smaller element
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Function to implement Quicksort
void quickSort(vector<int> &arr, int low, int high) {
    if (low < high) {
        // pi is partitioning index, arr[p] is now at right place
        int pi = partition(arr, low, high);

        // Recursively sort elements before partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// Function to merge two sorted arrays
void merge(vector<int> &arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Function to implement parallel Quicksort using MPI
void parallelQuickSort(vector<int> &arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        int my_rank, comm_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

        int chunk_size = (high - low + 1) / comm_size;
        int remainder = (high - low + 1) % comm_size;
        int* recv_counts = new int[comm_size];
        int* displacements = new int[comm_size];

        for (int i = 0; i < comm_size; i++) {
            recv_counts[i] = chunk_size;
            if (remainder > 0) {
                recv_counts[i]++;
                remainder--;
            }
            displacements[i] = i == 0 ? low : displacements[i - 1] + recv_counts[i - 1];
        }

        vector<int> local_arr(recv_counts[my_rank]);
        MPI_Scatterv(arr.data(), recv_counts, displacements, MPI_INT, local_arr.data(), recv_counts[my_rank], MPI_INT, 0, MPI_COMM_WORLD);

        int local_pi = partition(local_arr, 0, recv_counts[my_rank] - 1);

        vector<int> all_pivots(comm_size);
        MPI_Allgather(&local_arr[local_pi], 1, MPI_INT, all_pivots.data(), 1, MPI_INT, MPI_COMM_WORLD);

        int pivot_index = (comm_size - 1) / 2;
        int pivot = all_pivots[pivot_index];

        vector<int> lt, gt, eq;
        for (int i = 0; i < recv_counts[my_rank]; i++) {
            if (local_arr[i] < pivot)
                lt.push_back(local_arr[i]);
            else if (local_arr[i] > pivot)
                gt.push_back(local_arr[i]);
            else
                eq.push_back(local_arr[i]);
        }

        vector<int> all_lt, all_gt, all_eq;
        int* send_counts = new int[comm_size];
        int* recv_sizes = new int[comm_size];

        for (int i = 0; i < comm_size; i++) {
            send_counts[i] = 0;
            recv_sizes[i] = 0;
        }

        for (int i = 0; i < comm_size; i++) {
            MPI_Gather(&lt[0], lt.size(), MPI_INT, NULL, 0, MPI_INT, i, MPI_COMM_WORLD);
            MPI_Gather(&gt[0], gt.size(), MPI_INT, NULL, 0, MPI_INT, i, MPI_COMM_WORLD);
            MPI_Gather(&eq[0], eq.size(), MPI_INT, NULL, 0, MPI_INT, i, MPI_COMM_WORLD);
        }

        for (int i = 0; i < comm_size; i++) {
            send_counts[i] = lt.size();
            MPI_Scatter(send_counts, 1, MPI_INT, &recv_sizes[i], 1, MPI_INT, i, MPI_COMM_WORLD);
        }

        int total_lt = 0, total_eq = 0;
        for (int i = 0; i < comm_size; i++) {
            total_lt += recv_sizes[i];
            total_eq += eq.size();
        }

        delete[] recv_counts;
        delete[] displacements;
        delete[] send_counts;
        delete[] recv_sizes;

        vector<int> new_arr(total_lt + total_eq);

        MPI_Allgather(&total_lt, 1, MPI_INT, recv_sizes, 1, MPI_INT, MPI_COMM_WORLD);

        displacements[0] = 0;
        for (int i = 1; i < comm_size; i++)
            displacements[i] = displacements[i - 1] + recv_sizes[i - 1];

        MPI_Gatherv(&lt[0], lt.size(), MPI_INT, new_arr.data(), recv_sizes, displacements, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&eq[0], eq.size(), MPI_INT, new_arr.data(), recv_sizes, displacements, MPI_INT, 0, MPI_COMM_WORLD);

        if (my_rank == 0) {
            vector<int> lt_gt(all_gt.size());
            for (int i = 0; i < all_gt.size(); i++) {
                lt_gt[i] = all_gt[i];
            }

            quickSort(lt_gt, 0, lt_gt.size() - 1);

            for (int i = 0; i < new_arr.size(); i++) {
                cout << new_arr[i] << " ";
            }
            cout << endl;

            for (int i = 0; i < lt

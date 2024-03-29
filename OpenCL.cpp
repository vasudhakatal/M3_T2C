#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <ctime>

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

        // Separately sort elements before partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    // Generating random input
    srand(time(0));
    vector<int> arr;
    int n = 10000; // Adjust the size of the array as per your requirement
    for (int i = 0; i < n; i++) {
        arr.push_back(rand() % 1000);
    }

    // Printing unsorted array
    cout << "Unsorted array:" << endl;
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    // Measuring execution time
    clock_t start = clock();

    // Sorting the array
    quickSort(arr, 0, n - 1);

    // Measuring execution time
    clock_t end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);

    // Printing sorted array
    cout << "Sorted array:" << endl;
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    // Printing execution time
    cout << "Time taken by QuickSort: " << time_taken << " seconds" << endl;

    return 0;
}

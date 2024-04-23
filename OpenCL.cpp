#include <mpi.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include <string.h>
#define MAX_SIZE 1000000 // Maximum size of the array

// Function to swap two elements
void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Partition function for Quicksort
int partition(int arr[], int low, int high)
{
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++)
    {
        if (arr[j] < pivot)
        {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

// Quicksort function
void quicksort(int arr[], int low, int high)
{
    if (low < high)
    {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

// Parallel Quicksort using MPI and OpenCL
void parallel_quicksort(int arr[], int n, int rank, int size)
{
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem mem_obj;

    // Get platform and device information
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    // Create OpenCL context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create command queue
    command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);

    // Create memory buffer
    mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * n, NULL, &ret);

    // Write data to the memory buffer
    ret = clEnqueueWriteBuffer(command_queue, mem_obj, CL_TRUE, 0, sizeof(int) * n, arr, 0, NULL, NULL);

    // Create OpenCL program from source
    const char *source_str =
        "_kernel void quicksort(_global int* arr, int low, int high) {"
        "    if (low < high) {"
        "        int pivot = arr[high];"
        "        int i = low - 1;"
        "        for (int j = low; j <= high - 1; j++) {"
        "            if (arr[j] < pivot) {"
        "                i++;"
        "                int temp = arr[i];"
        "                arr[i] = arr[j];"
        "                arr[j] = temp;"
        "            }"
        "        }"
        "        int temp = arr[i + 1];"
        "        arr[i + 1] = arr[high];"
        "        arr[high] = temp;"
        ""
        "        int pi = i + 1;"
        "        quicksort(arr, low, pi - 1);"
        "        quicksort(arr, pi + 1, high);"
        "    }"
        "}";

    size_t source_size = strlen(source_str);
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build OpenCL program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "quicksort", &ret);

    // Set OpenCL kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(int), (void *)&arr[0]);
    ret = clSetKernelArg(kernel, 2, sizeof(int), (void *)&arr[n - 1]);

    // Execute OpenCL kernel
    size_t global_item_size = n;
    size_t local_item_size = 1;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    // Read the memory buffer
    ret = clEnqueueReadBuffer(command_queue, mem_obj, CL_TRUE, 0, sizeof(int) * n, arr, 0, NULL, NULL);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Test case
    int arr[MAX_SIZE];
    int n = MAX_SIZE;
    srand(time(NULL));
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand(); // Generate random numbers between 0 and 999999
    }

    double start_time, end_time;
    start_time = MPI_Wtime();
    parallel_quicksort(arr, n, rank, size);
    end_time = MPI_Wtime();

    if (rank == 0)
    {
        double execution_time_us = (end_time - start_time) * 1000000;
        printf("Time Taken: %.0f microseconds\n", execution_time_us);
    }

    MPI_Finalize();
    return 0;
}

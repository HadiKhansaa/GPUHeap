#include <iostream>
#include <time.h>
#include <algorithm>
#include <functional>
#include <queue>
#include "header/heap.cuh" 
#include "header/MinHeap.h"
#include "header/timer.h" 

#define INSERT true  
#define DELETE false

struct inputEntry {
    bool operation;
    int* elements;
    int size;
};

// Function to generate a random array of given size
int* generateRandomArray(int size) {
    int* arr = new int[size];
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 100; // generating random numbers between 0 and 99
    }
    return arr;
}

// assigns each block to a batch of insert operations, blockIdx.x = batchIdx
__global__ void insertKernel(Heap *heap, int *items, int arraySize, int batchSize) {
    int batchNeed = (arraySize + batchSize - 1) / batchSize;
    // No more blocks needed
    if(blockIdx.x >= batchNeed) return;
    // Each thread in block inserts its block to the heap in parallel
    heap->insertion(items + blockIdx.x * batchSize, batchSize, 0);
    __syncthreads();
}

__global__ void deleteKernel(Heap *heap, int *items, int arraySize, int batchSize) {
    uint32_t batchNeed = arraySize / batchSize;
    if(blockIdx.x >= batchNeed) return;
    int size = 0;
    // delete items from heap
    if (heap->deleteRoot(items, size)) {
        __syncthreads();
        heap->deleteUpdate(0);
    }
    __syncthreads();
    
}

int main(int argc, char *argv[]) {

    // change the seed
    srand(time(NULL));
    
    Timer timer;

    // Block size and number of blocks
    int blockSize = 4;

    //size of the array we need to insert
    int arrayNum = 1000;

    if(argc == 3)
    {
        blockSize = atoi(argv[1]);
        arrayNum = atoi(argv[2]);
    }

    int batchSize = blockSize;
    arrayNum = (arrayNum + batchSize - 1) / batchSize * batchSize;

    // Batch size and number of batches
    int batchNum = (arrayNum + batchSize - 1) / batchSize;
    int blockNum = batchNum;

    // generate random array for testing
    int *h_tItems = new int[arrayNum];
    for (int i = 0; i < arrayNum; ++i) {
        h_tItems[i] = rand() % 100;
    }

    // Sort the array
    // std::sort(h_tItems, h_tItems + arrayNum);

    MinHeap<int> h;

    startTime(&timer);

    for (int i = 0; i < arrayNum; ++i) {
        h.insertNode(h_tItems[i]);
    }
    stopTime(&timer);
    printElapsedTime(timer, "CPU Sequential Heap", CYAN);

    // test std::priority_queue
    std::priority_queue<int, std::vector<int>, std::greater<int>> pq;
    startTime(&timer);
    for (int i = 0; i < arrayNum; ++i) {
        pq.push(h_tItems[i]);
    }
    stopTime(&timer);
    printElapsedTime(timer, "std Priority Queue", ORANGE);



    // Heap construction
    Heap h_heap(batchNum, batchSize);

    // Create a heap on device
    int *heapItems;
    Heap *d_heap;

    // Allocate momory and copy on device
    cudaMalloc((void **)&heapItems, sizeof(int) * (arrayNum));
    cudaMemcpy(heapItems, h_tItems, sizeof(int) * (arrayNum), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_heap, sizeof(Heap));
    cudaMemcpy(d_heap, &h_heap, sizeof(Heap), cudaMemcpyHostToDevice);

    // Allocate shared memory
    int smemSize = batchSize * 3 * sizeof(int);
    smemSize += (blockSize + 1) * sizeof(int) + 2 * batchSize * sizeof(int);

    // call the kernel 
    startTime(&timer);
    insertKernel<<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, arrayNum, batchSize);

    cudaDeviceSynchronize();

    // Error checking
    auto err = cudaGetLastError();
    if(err != cudaSuccess)
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;

    stopTime(&timer);
    printElapsedTime(timer, "Parrallel Heap Insert", DGREEN);

    // validate the heap
    if(h_heap.checintInsertHeap())
        std::cout << "\033[1;32m" << "Heap is valid" << std::endl << "\033[0m";
    else
        std::cout << "\033[0;31m" << "Heap is invalid"<< std::endl << "\033[0m";


    // deletion kernel

    startTime(&timer);

    deleteKernel<<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, arrayNum, batchSize);

    cudaDeviceSynchronize();

    // Error checking
    err = cudaGetLastError();
    if(err != cudaSuccess)
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;

    stopTime(&timer);
    printElapsedTime(timer, "Parrallel Heap Delete", GREEN);
    
    delete []h_tItems;

    // h_heap.printHeap();
    
}

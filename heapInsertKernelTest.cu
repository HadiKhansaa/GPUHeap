#include <iostream>
#include <algorithm>
#include <functional>
#include "heap.cuh" 
#include "MinHeap.h" 
#include "timer.h" 

#define INSERT true  
#define DELETE false

template<typename K = int>
struct inputEntry {
    bool operation;
    K* elements;
    int size;
};

// Function to generate a random array of given size
template<typename K = int>
K* generateRandomArray(int size) {
    K* arr = new K[size];
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 100; // generating random numbers between 0 and 99
    }
    return arr;
}

// assigns each block to a batch of insert operations, blockIdx.x = batchIdx
__global__ void insertKernel(Heap<int> *heap, int *items, int arraySize, int batchSize) {
    int batchNeed = arraySize / batchSize;
    // No more blocks needed
    if(blockIdx.x >= batchNeed || threadIdx.x > batchSize) return;
    // Each thread in block inserts its block to the heap in parallel
    heap->insertion(items + blockIdx.x * batchSize, batchSize, 0);
    __syncthreads();
    
}

int main(int argc, char *argv[]) {
    Timer timer;

    // Arguments
    if (argc != 5) {
        std::cout << argv[0] << " [test type] [# init in M] [# keys in M] [keyType: 0: random 1: ascend 2: descend]\n";
        return -1;
    }

    

    int testType = atoi(argv[1]); // Insert/delete/concurrent
    int arrayNum = atoi(argv[3]); // Size of the array we need to insert

    // Batch size and number of batches
    int batchSize = 128;
    int batchNum = (arrayNum + batchSize - 1) / batchSize;

    // Block size and number of blocks
    int blockSize = 128;
    int blockNum = batchNum;

    arrayNum = (arrayNum + batchSize - 1) / batchSize * batchSize;
    int keyType = atoi(argv[4]); // type of keys to insert // 0: random, 1: ascending, 2: descending

    // generate random array for testing
    int *h_tItems = new int[arrayNum];
    for (int i = 0; i < arrayNum; ++i) {
        h_tItems[i] =rand() % INT_MAX;
    }

    // Sort the array
    std::sort(h_tItems, h_tItems + arrayNum);

    MinHeap<int> h;

    startTime(&timer);

    for (int i = 0; i < arrayNum; ++i) {
        h.insertNode(h_tItems[i]);
    }
    stopTime(&timer);
    printElapsedTime(timer, "Sequential Heap", DGREEN);


    // Heap construction
    Heap<int> h_heap(batchNum, batchSize, INT_MAX);

    // Create a heap on device
    int *heapItems;
    Heap<int> *d_heap;

    // Allocate momory and copy on device
    cudaMalloc((void **)&heapItems, sizeof(int) * (arrayNum));
    cudaMemcpy(heapItems, h_tItems, sizeof(int) * (arrayNum), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_heap, sizeof(Heap<int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(Heap<int>), cudaMemcpyHostToDevice);

    // Allocate shared memory
    int smemSize = batchSize * 3 * sizeof(int);
    smemSize += (blockSize + 1) * sizeof(int) + 2 * batchSize * sizeof(int);

    if (testType == 0) { // Insertion
        startTime(&timer);
        insertKernel<<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, arrayNum, batchSize);
        stopTime(&timer);
        printElapsedTime(timer, "Parrallel Heap", CYAN);
        delete []h_tItems;
    }
    // Print the heap after insertion
    // h_heap.printHeap();
}

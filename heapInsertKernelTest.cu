#include <iostream>
#include <time.h>
#include <algorithm>
#include <functional>
#include <queue>
#include "heap.cuh" 
#include "MinHeap.h"
#include "timer.h" 

#define INSERT true  
#define DELETE false

__global__ void concurrentKernel(Heap *heap, inputEntry *entries, int entriesSize, int nodeSize) {
    int batchNeed = entriesSize / nodeSize;
    // insertion
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x) {
        // insert items to buffer
        if (entries[i].operation == INSERT) {
            heap->insertion(entries[i].elements, entries[i].size, 0);
        } else {
            heap->deleteRoot(entries[i].elements, entries[i].size);
        }
        __syncthreads();
    }
}

__global__ void insertKernel(Heap *heap, int *items, int arraySize, int nodeSize) {
    int batchNeed = arraySize / nodeSize;
    // insertion
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x) {
        // insert items to buffer
        heap->insertion(items + i * nodeSize, nodeSize, 0);
        __syncthreads();
    }
}

__global__ void deleteKernel(Heap *heap, int *items, int arraySize, int nodeSize) {
    int batchNeed = arraySize / nodeSize;
    int size = nodeSize;
    // deletion
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x) {
        // delete items from heap
        if (heap->deleteRoot(items, size) == true) {
            __syncthreads();
            heap->deleteUpdate(0);
        }
        __syncthreads();
    }
}

void testSeqHeap(int *h_tItems, int arraySize){
    Timer timer;
    MinHeap<int> h;
    startTime(&timer);
    for (int i = 0; i < arraySize; ++i) {
        h.insertNode(h_tItems[i]);
    }
    stopTime(&timer);
    printElapsedTime(timer, "CPU Sequential Heap", CYAN);
}

void testPriorityQueue(int *h_tItems, int arraySize){
    Timer timer;
    std::priority_queue<int, std::vector<int>, std::greater<int>> pq;
    startTime(&timer);
    for (int i = 0; i < arraySize; ++i) {
        pq.push(h_tItems[i]);
    }
    stopTime(&timer);
    printElapsedTime(timer, "std Priority Queue", ORANGE);
}

void testGpuHeap(int *h_tItems, int numThreadsPerBlock, int nodeSize, int arraySize){
    Timer timer;

    // Batch size and number of batches
    int maxNumNodes = (arraySize + nodeSize - 1) / nodeSize;
    int numBlocks = maxNumNodes;
    Heap h_heap(maxNumNodes, nodeSize);

    // Create a heap on device
    int *heapItems;
    Heap *d_heap;

    // Allocate momory and copy on device
    cudaMalloc((void **)&heapItems, sizeof(int) * (arraySize));
    cudaMemcpy(heapItems, h_tItems, sizeof(int) * (arraySize), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_heap, sizeof(Heap));
    cudaMemcpy(d_heap, &h_heap, sizeof(Heap), cudaMemcpyHostToDevice);

    // Allocate shared memory
    int smemSize = nodeSize * 3 * sizeof(int);
    smemSize += (numThreadsPerBlock + 1) * sizeof(int) + 2 * nodeSize * sizeof(int);

    // call the kernel 
    startTime(&timer);
    insertKernel<<<numBlocks, numThreadsPerBlock, smemSize>>>(d_heap, heapItems, arraySize, nodeSize);

    cudaDeviceSynchronize();

    // Error checking
    auto err = cudaGetLastError();
    if(err != cudaSuccess)
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;

    stopTime(&timer);

    // int *heapItems_deleted;
    // int arraySizeDeleted = 2048;
    // // Allocate momory and copy on device
    // cudaMalloc((void **)&heapItems_deleted, sizeof(int) * (arraySizeDeleted));
    // deleteKernel<<<numBlocks, numThreadsPerBlock, smemSize>>>(d_heap, heapItems_deleted, arraySizeDeleted , nodeSize);
    
    // Error checking
    // auto err = cudaGetLastError();
    // if(err != cudaSuccess)
    //     std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;

    
    printElapsedTime(timer, "Parrallel Heap Insert", DGREEN);

    // validate the heap
    if(h_heap.checkintInsertHeap())
        std::cout << "\033[1;32m" << "Heap is valid" << std::endl << "\033[0m";
    else
        std::cout << "\033[0;31m" << "Heap is invalid"<< std::endl << "\033[0m";

    delete []h_tItems;
    // h_heap.printHeap();
}

int main(int argc, char *argv[]) {

    // change the seed
    srand(time(NULL));

    // Block size and batch size
    int numThreadsPerBlock = 512;
    int nodeSize = 1024;

    //size of the array we need to insert
    int arraySize = 1000;

    if(argc == 4) {
        numThreadsPerBlock = atoi(argv[1]);
        nodeSize = atoi(argv[2]);
        arraySize = atoi(argv[3]);
    }
    else {
        std::cout << "Usage: ./heapInsertKernelTest <numThreadsPerBlock> <nodeSize> <arraySize>" << std::endl;
        return 1;
    }
    arraySize = ((arraySize + nodeSize - 1)/nodeSize) * nodeSize;

    // generate random array for testing
    int* h_tItems = generateRandomArray(arraySize);

    // test Sequentioal MinHeap
    testSeqHeap(h_tItems,arraySize);

    // test std::priority_queue
    testPriorityQueue(h_tItems,arraySize);

    // Heap construction
    testGpuHeap(h_tItems, numThreadsPerBlock, nodeSize, arraySize);
}
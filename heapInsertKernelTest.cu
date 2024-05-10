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

    // time copy to device
    startTime(&timer);

    // Allocate momory and copy on device
    gpuErrchk(cudaMalloc((void **)&heapItems, sizeof(int) * (arraySize)));
    gpuErrchk(cudaMemcpy(heapItems, h_tItems, sizeof(int) * (arraySize), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void **)&d_heap, sizeof(Heap)));
    gpuErrchk(cudaMemcpy(d_heap, &h_heap, sizeof(Heap), cudaMemcpyHostToDevice));

    stopTime(&timer);
    printElapsedTime(timer, "Copy to Device", RED);

    // Allocate shared memory
    int smemSize = nodeSize * 3 * sizeof(int);
    smemSize += (numThreadsPerBlock + 1) * sizeof(int) + 2 * nodeSize * sizeof(int);

    // call the kernel 
    startTime(&timer);
    insertKernel<<<numBlocks, numThreadsPerBlock, smemSize>>>(d_heap, heapItems, arraySize, nodeSize);

    gpuErrchk(cudaDeviceSynchronize());

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

void testGpuHeapMultipleStrams() {
    // Number of streams
    const int NUM_STREAMS = 5;

    // Your existing setup here
    int *heapItems, *d_heapItems[NUM_STREAMS];
    Heap *d_heap[NUM_STREAMS];
    int arraySize = 10000000;  // Example size
    int nodeSize = 2048;     // Example node size
    int blockSize = 512;

    int numBlocks = (arraySize + blockSize - 1) / blockSize;
    int maxNumNodes = (arraySize + nodeSize - 1) / nodeSize;

    Heap h_heap(maxNumNodes, nodeSize);

    // Stream array
    cudaStream_t streams[NUM_STREAMS];

    // Initialize streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        gpuErrchk(cudaStreamCreate(&streams[i]));
    }

    // Divide your work into parts for each stream
    int chunkSize = arraySize / NUM_STREAMS;

    // Initialize and allocate memory for host data (if not already done)
    int *h_tItems = new int[arraySize];

    // Fill your host data with values (example)
    for (int i = 0; i < arraySize; ++i) {
        h_tItems[i] = rand() % 100;  // Example data
    }

    Timer timer;
    startTime(&timer);
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunkSize;
        int currentChunkSize = (i == NUM_STREAMS - 1) ? (arraySize - offset) : chunkSize;

        // Allocate and copy data in streams
        gpuErrchk(cudaMalloc((void **)&d_heapItems[i], sizeof(int) * currentChunkSize));
        gpuErrchk(cudaMemcpyAsync(d_heapItems[i], h_tItems + offset, sizeof(int) * currentChunkSize, cudaMemcpyHostToDevice, streams[i]));

        gpuErrchk(cudaMalloc((void **)&d_heap[i], sizeof(Heap)));
        gpuErrchk(cudaMemcpyAsync(d_heap[i], &h_heap, sizeof(Heap), cudaMemcpyHostToDevice, streams[i]));

        // Allocate shared memory size based on your existing setup
        int smemSize = nodeSize * 3 * sizeof(int);
        smemSize += (blockSize + 1) * sizeof(int) + 2 * nodeSize * sizeof(int);

        // Call the kernel in streams
        insertKernel<<<numBlocks, blockSize, smemSize, streams[i]>>>(d_heap[i], d_heapItems[i], currentChunkSize, nodeSize);

        // std::cout<<"Stream "<<i<<" launched"<<std::endl;
    }

    // Wait for all streams to complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        gpuErrchk(cudaStreamSynchronize(streams[i]));
        gpuErrchk(cudaStreamDestroy(streams[i]));
        gpuErrchk(cudaFree(d_heapItems[i]));
        gpuErrchk(cudaFree(d_heap[i]));
    }

    stopTime(&timer);
    printElapsedTime(timer, "Parrallel Heap Insert", DGREEN);

    // validate the heap
    if(h_heap.checkintInsertHeap())
        std::cout << "\033[1;32m" << "Heap is valid" << std::endl << "\033[0m";
    else
        std::cout << "\033[0;31m" << "Heap is invalid"<< std::endl << "\033[0m";

    // print heap
    // h_heap.printHeap();

    // Clean up host memory
    delete[] h_tItems;
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
    // testGpuHeap(h_tItems, numThreadsPerBlock, nodeSize, arraySize);

    testGpuHeapMultipleStrams();
}
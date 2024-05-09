#ifndef HEAPUTIL_CUH
#define HEAPUTIL_CUH

struct inputEntry {
    bool operation;
    int* elements;
    int size;
};

#include <unordered_set>
// Atomic operation to change the status of a node from origninal state to new state.
        // int atomicCAS(int* address, int compare, int val); // atomic compare and swap
        // returns the initial value pointed to by address.
        __device__ bool changeStatus(int *_status, int oriS, int newS) {
            if ((oriS == AVAIL  && newS == TARGET) ||
                (oriS == TARGET && newS == MARKED) ||
                (oriS == MARKED && newS == TARGET) ||
                (oriS == TARGET && newS == AVAIL ) ||
                (oriS == TARGET && newS == INUSE ) ||
                (oriS == INUSE  && newS == AVAIL ) ||
                (oriS == AVAIL  && newS == INUSE )) {
                while (atomicCAS(_status, oriS, newS) != oriS){
                }
                return true;
            }
            else {
                printf("LOCK ERROR %d %d\n", oriS, newS);
                return false;
            }
        }

        // determine the next batch when insert operation updating the heap
        // given the current batch index and the target batch index
        // return the next batch index to the target batch
        // __clz (Count Leading Zeros) function calculates the number of leading zeros in the binary representation. 
         __device__ int getNextIdxToTarget(int currentIdx, int targetIdx) {
            return targetIdx >> (__clz(currentIdx) - __clz(targetIdx) - 1);
        }


__inline__ __device__ void batchCopy(int  *dest, int  *source, int size, bool reset = false, int  init_limits = 0)
{
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        dest[i] = source[i];
        if (reset) source[i] = init_limits;
    }
    __syncthreads();
}


__inline__ __device__ void _swap(int &a, int &b) {
    int tmp = a;
    a = b;
    b = tmp;
}

__inline__ __device__ void ibitonicSort(int  *items, int size) {

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k / 2; j > 0; j >>= 1) {
            for (int i =  threadIdx.x; i < size; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (items[i] > items[ixj]) {
                            _swap(items[i], items[ixj]);
                        }
                    }
                    else {
                        if (items[i] < items[ixj]) {
                            _swap(items[i], items[ixj]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

__inline__ __device__ void imergePath(int *aItems, int *bItems,
                                      int *smallItems, int *largeItems,
                                      int size, int smemOffset) {

    extern __shared__ int s[];
    int *tmpItems = (int *)&s[smemOffset];

    int lengthPerThread = size * 2 / blockDim.x;

    int index= threadIdx.x * lengthPerThread;
    int aTop = (index > size) ? size : index;
    int bTop = (index > size) ? index - size : 0;
    int aBottom = bTop;
    
    int offset, aI, bI;
    
    // binary search for diagonal intersections
    while (1) {
        offset = (aTop - aBottom) / 2;
        aI = aTop - offset;
        bI = bTop + offset;

        if (aTop == aBottom || (bI < size && (aI == size || aItems[aI] > bItems[bI]))) {
            if (aTop == aBottom || aItems[aI - 1] <= bItems[bI]) {
                break;
            }
            else {
                aTop = aI - 1;
                bTop = bI + 1;
            }
        }
        else {
            aBottom = aI;
        }
     }

     // start from [aI, bI], found a path with lengthPerThread
    for (int i = lengthPerThread * threadIdx.x; i < lengthPerThread * threadIdx.x + lengthPerThread; ++i) {
        if (bI == size || (aI < size && aItems[aI] <= bItems[bI])) {
            tmpItems[i] = aItems[aI];
            aI++;
        }
        else if (aI == size || (bI < size && aItems[aI] > bItems[bI])) {
            tmpItems[i] = bItems[bI];
            bI++;
        }
    }
    __syncthreads();

    batchCopy(smallItems, tmpItems, size);
    batchCopy(largeItems, tmpItems + size, size);
}

// Function to generate a random array
int* generateRandomArray(int arraySize) {
    // Seed the random number generator
    srand(time(nullptr));

    int* randomArray = new int[arraySize];

    for (int i = 0; i < arraySize; ++i) {
        randomArray[i] = rand() % 10000; // Generate random number between 0 and 9999
    }
    return randomArray;
}

inputEntry* generateRandomInput(int arraySize, int nodeSize) {
    inputEntry* randomInput = new inputEntry[arraySize / nodeSize];
    for (int i = 0; i < arraySize / nodeSize; ++i) {
        randomInput[i].operation = rand() % 2;
        randomInput[i].elements = new int[nodeSize];
        randomInput[i].size = nodeSize;
        for (int j = 0; j < nodeSize; ++j) {
            randomInput[i].elements[j] = rand() % 10000;
        }
    }
    return randomInput;
}

inputEntry* flatenInput (inputEntry* input, int arraySize, int nodeSize) {
    inputEntry* flatenInput = new inputEntry[arraySize];
    for (int i = 0; i < arraySize / nodeSize; ++i) {
        for (int j = 0; j < nodeSize; ++j) {
            flatenInput[i * nodeSize + j] = input[i];
        }
    }
    return flatenInput;
}

#endif

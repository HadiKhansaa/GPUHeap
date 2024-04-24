#ifndef HEAPUTIL_CUH
#define HEAPUTIL_CUH

#include <unordered_set>

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

#endif

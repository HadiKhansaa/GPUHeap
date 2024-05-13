#ifndef HEAP_CUH
#define HEAP_CUH

#define AVAIL 0
#define INUSE 1
#define TARGET 2
#define MARKED 3

#include "heaputil.cuh"


class Heap {
    public:
        // Number of batches (ndoes) in the Heap 
        int maxNumNodes;
        // Size of each batch (node) in the Heap 
        int nodeSize;
        // How many batch is in use 
        int *currNumNodes;
        // How many elements is in th buffer
        int *partialBufferSize;
        // Array of the heap of size nodeSize * (maxNumNodes + 1)
        int *heapItems;
        // Array of size maxNumNodes, each element in this array correspond for a node status in the Heap
        int *status;

    // constructor 
    Heap(int _maxNumNodes, int _nodeSize) : maxNumNodes(_maxNumNodes), nodeSize(_nodeSize) {
        // prepare device heap
        cudaMalloc((void **)&heapItems, sizeof(int) * nodeSize * (maxNumNodes + 1));

        // initialize heap items with max value
        int *tmp = new int[nodeSize * (maxNumNodes + 1)];
        std::fill(tmp, tmp + nodeSize * (maxNumNodes + 1), INT_MAX);
        cudaMemcpy(heapItems, tmp, sizeof(int) * nodeSize * (maxNumNodes + 1), cudaMemcpyHostToDevice);
        delete []tmp; tmp = NULL;

        // initialize all the nodes status to Available (AVAIL)
        cudaMalloc((void **)&status, sizeof(int) * (maxNumNodes + 1));
        cudaMemset(status, AVAIL, sizeof(int) * (maxNumNodes + 1));

        // initialize currNumNodes by 0 
        cudaMalloc((void **)&currNumNodes, sizeof(int));
        cudaMemset(currNumNodes, 0, sizeof(int));

        // initialize partialBufferSize by 0 
        cudaMalloc((void **)&partialBufferSize, sizeof(int));
        cudaMemset(partialBufferSize, 0, sizeof(int));
    }
    
    // Reset the Heap to the initial state
    void reset() {
        int *tmp = new int[nodeSize * (maxNumNodes + 1)];
        for (int i = 0; i < (maxNumNodes + 1) * nodeSize; i++) {
            tmp[i] = INT_MAX;
        }
        cudaMemcpy(heapItems, tmp, sizeof(int) * nodeSize * (maxNumNodes + 1), cudaMemcpyHostToDevice);
        delete []tmp; tmp = NULL;

        cudaMemset(status, AVAIL, sizeof(int) * (maxNumNodes + 1));
        cudaMemset(currNumNodes, 0, sizeof(int));
        cudaMemset(partialBufferSize, 0, sizeof(int));
    }

    // Destructor 
    ~Heap() {
        cudaFree(heapItems);
        heapItems = NULL;
        cudaFree(status);
        status = NULL;
        cudaFree(currNumNodes);
        currNumNodes = NULL;
        cudaFree(partialBufferSize);
        partialBufferSize = NULL;
        maxNumNodes = 0;
        nodeSize = 0;
    }

    // checks the validity for the heap
    bool checkintInsertHeap() {
        int h_currNumNodes;
        int h_partialBufferSize;
        cudaMemcpy(&h_currNumNodes, currNumNodes, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_partialBufferSize, partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);

        int *h_status = new int[h_currNumNodes + 1];
        int *h_items = new int[nodeSize * (h_currNumNodes + 1)];
        cudaMemcpy(h_items, heapItems, sizeof(int) * nodeSize * (h_currNumNodes + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_status, status, sizeof(int) * (h_currNumNodes + 1), cudaMemcpyDeviceToHost);

        // create a bug in heap
        
        // check partial batch
        if (h_status[0] != AVAIL) {
            printf("Partial Batch State Error: state should be AVAIL = 0 while current is %d\n", h_status[0]);
            return false;
        }
        if (h_currNumNodes != 0 && h_partialBufferSize != 0) {
            if (h_items[nodeSize * 2 - 1] > h_items[0]) {
                printf("Partial Buffer Error: partial batch should be larger than root batch.\n");
                return false;
            }
            for (int i = 1; i < h_partialBufferSize; i++) {
                if (h_items[i] < h_items[i - 1]) {
                    printf("Partial Buffer Error: partialBuffer[%d] is smaller than partialBuffer[%d-1]\n", i, i); 
                    return false;
                }
            }
        }

        for (int i = 1; i <= h_currNumNodes; ++i) {
            if (h_status[i] != AVAIL) {
                printf("State Error @ batch %d, state should be AVAIL = 0 while current is %d\n", i, h_status[i]);
                return false;
            }
            if (i > 1) {
                if (h_items[i * nodeSize] < h_items[i/2 * nodeSize + nodeSize - 1]){
                    printf("Batch Keys Error @ batch %d's first item is smaller than batch %d's last item\n", i, i/2);
                    return false;
                }
            }
            for (int j = 1; j < nodeSize; ++j) {
                if (h_items[i * nodeSize + j] < h_items[i * nodeSize + j - 1]) {
                    printf("Batch Keys Error @ batch %d item[%d] is smaller than item[%d]\n", i, j, j - 1);
                    return false;
                }
            }
        }

        delete []h_items;
        delete []h_status;

        return true;
    }
        

    // print the Heap nodes, each node of size nodeSize at a line 
    void printHeap() {
        // copy the heap fields to the host 
        int h_currNumNodes;
        int h_partialBufferSize;
        cudaMemcpy(&h_currNumNodes, currNumNodes, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_partialBufferSize, partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);

        int *h_status = new int[h_currNumNodes + 1];
        int *h_items = new int[nodeSize * (h_currNumNodes + 1)];
        cudaMemcpy(h_items, heapItems, sizeof(int) * nodeSize * (h_currNumNodes + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_status, status, sizeof(int) * (h_currNumNodes + 1), cudaMemcpyDeviceToHost);

        // printing the buffer
        printf("batch partial %d_%d:", h_partialBufferSize, h_status[0]);

        for (int i = 0; i < h_partialBufferSize; ++i) {
            printf(" %d", h_items[i]);
        }
        printf("\n");

        //printing the batches (nodes)
        for (int i = 1; i <= h_currNumNodes; ++i) {
            printf("batch %d_%d:", i, h_status[i]);
            for (int j = 0; j < nodeSize; ++j) {
                printf(" %d", h_items[i * nodeSize + j]);
            }
            printf("\n");
        }
    }

    // Returns the total number of items in the heap (on the host).
    int itemCount() {
        int psize, bcount;
        cudaMemcpy(&bcount, currNumNodes, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&psize, partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);
        return psize + bcount * nodeSize;
    }

    // Checks if the heap is empty.
    __host__ bool isEmpty() {
        int psize, bsize;
        cudaMemcpy(&psize, partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bsize, currNumNodes, sizeof(int), cudaMemcpyDeviceToHost);
        return !psize && !bsize;
    }

    __device__ void insertion(int *items, int size, int smemOffset) {
        // allocate shared memory space
        extern __shared__ int s[]; // dynamically allocate shared memory per block. 
        int *sMergedItems1 = (int *)&s[smemOffset];
        int *sMergedItems2 = (int *)&sMergedItems1[nodeSize];
        smemOffset += 2 * nodeSize;
        int *tmpIdx = (int *)&s[smemOffset - 1];

        // move insert batch to shared memory
        // may be a partial batch, fill rest part with INT_MAX
        for (int i = threadIdx.x; i < nodeSize; i += blockDim.x) {
            sMergedItems1[i] = i < size ? items[i] : INT_MAX;
        }
        __syncthreads();
        
        ibitonicSort(sMergedItems1, nodeSize);
        __syncthreads();

        if (threadIdx.x == 0) {
            changeStatus(&status[0], AVAIL, INUSE);
        }
        __syncthreads();

        /* start handling partial batch */
        // Case 1: the heap has no full batch
        if (*currNumNodes == 0 && size < nodeSize) {
            // Case 1.1: partial batch is empty
            if (*partialBufferSize == 0) {
                batchCopy(heapItems, sMergedItems1, nodeSize);
                if (threadIdx.x == 0) {
                    *partialBufferSize = size;
                    changeStatus(&status[0], INUSE, AVAIL);
                }
                __syncthreads();
                return;
            }
            // Case 1.2: no full batch is generated
            else if (size + *partialBufferSize < nodeSize) {

                batchCopy(sMergedItems2, heapItems, nodeSize);
                imergePath(sMergedItems1, sMergedItems2, heapItems, sMergedItems1, nodeSize, smemOffset);
                __syncthreads();
                if (threadIdx.x == 0) {
                    *partialBufferSize += size;
                    changeStatus(&status[0], INUSE, AVAIL);
                }
                __syncthreads();
                return;
            }
            // Case 1.3: a full batch is generated
            else if (size + *partialBufferSize >= nodeSize) {
                batchCopy(sMergedItems2, heapItems, nodeSize);
                if (threadIdx.x == 0) {
                    // increase currNumNodes and change root batch to INUSE
                    atomicAdd(currNumNodes, 1);
                    changeStatus(&status[1], AVAIL, TARGET);
                    changeStatus(&status[1], TARGET, INUSE);
                }
                __syncthreads();
                imergePath(sMergedItems1, sMergedItems2, heapItems + nodeSize, heapItems, nodeSize, smemOffset);
                __syncthreads();
                if (threadIdx.x == 0) {
                    *partialBufferSize += (size - nodeSize);
                    changeStatus(&status[0], INUSE, AVAIL);
                    changeStatus(&status[1], INUSE, AVAIL);
                }
                __syncthreads();
                return;
            }
        }
        // Case 2: the heap is non empty
        else {
            // Case 2.1: no full batch is generated
            if (size + *partialBufferSize < nodeSize) {
                batchCopy(sMergedItems2, heapItems, nodeSize);
                // Merge insert batch with partial batch
                imergePath(sMergedItems1, sMergedItems2,sMergedItems1, sMergedItems2,nodeSize, smemOffset);
                __syncthreads();
                if (threadIdx.x == 0) {
                    changeStatus(&status[1], AVAIL, INUSE);
                }
                __syncthreads();
                batchCopy(sMergedItems2, heapItems + nodeSize, nodeSize);
                imergePath(sMergedItems1, sMergedItems2, heapItems + nodeSize, heapItems, nodeSize, smemOffset);
                __syncthreads();
                if (threadIdx.x == 0) {
                    *partialBufferSize += size;
                    changeStatus(&status[0], INUSE, AVAIL);
                    changeStatus(&status[1], INUSE, AVAIL);
                }
                __syncthreads();
                return;
            }
            // Case 2.2: a full batch is generated and needed to be propogated
            else if (size + *partialBufferSize >= nodeSize) {
                batchCopy(sMergedItems2, heapItems, nodeSize);
                // Merge insert batch with partial batch, leave larger half in the partial batch
                imergePath(sMergedItems1, sMergedItems2, sMergedItems1, heapItems, nodeSize, smemOffset);
                __syncthreads();
                if (threadIdx.x == 0) {
                    // update partial batch size 
                    *partialBufferSize += (size - nodeSize);
                }
                __syncthreads();
            }
        }
        /* end handling partial batch */

        if (threadIdx.x == 0) {
            *tmpIdx = atomicAdd(currNumNodes, 1) + 1;
            changeStatus(&status[*tmpIdx], AVAIL, TARGET);
            if (*tmpIdx != 1) {
                changeStatus(&status[1], AVAIL, INUSE);
            }
        }
        __syncthreads();

        int currentIdx = 1;
        int targetIdx = *tmpIdx;
        __syncthreads();

        while(currentIdx != targetIdx) {
            if (threadIdx.x == 0) {
                *tmpIdx = 0;
                if (status[targetIdx] == MARKED) {
                    *tmpIdx = 1;
                }
            }
            __syncthreads();

            if (*tmpIdx == 1) break;
            __syncthreads();

            if (threadIdx.x == 0) {
                changeStatus(&status[currentIdx / 2], INUSE, AVAIL);
            }
            __syncthreads();

            // move batch to shard memory
            batchCopy(sMergedItems2, heapItems + currentIdx * nodeSize, nodeSize);

            if (sMergedItems1[nodeSize - 1] <= sMergedItems2[0]) {
                // if insert batch is smaller than current batch
                __syncthreads();
                batchCopy(heapItems + currentIdx * nodeSize, sMergedItems1, nodeSize);

                batchCopy(sMergedItems1, sMergedItems2, nodeSize);
            }
            else if (sMergedItems2[nodeSize - 1] > sMergedItems1[0]) {
                __syncthreads();
                imergePath(sMergedItems1, sMergedItems2, heapItems + currentIdx * nodeSize, sMergedItems1, nodeSize, smemOffset);
                __syncthreads();
            }
            currentIdx = getNextIdxToTarget(currentIdx, targetIdx);
            if (threadIdx.x == 0) {
                if (currentIdx != targetIdx) {
                    changeStatus(&status[currentIdx], AVAIL, INUSE);
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicCAS(&status[targetIdx], TARGET, INUSE);
        }
        __syncthreads();

        if (status[targetIdx] == MARKED) {
            __syncthreads();
            batchCopy(heapItems + nodeSize, sMergedItems1, nodeSize);
            if (threadIdx.x == 0) {
                changeStatus(&status[currentIdx / 2], INUSE, AVAIL);
                if (targetIdx != currentIdx) {
                    changeStatus(&status[currentIdx], INUSE, AVAIL);
                }
                changeStatus(&status[targetIdx], MARKED, TARGET);
            }
            __syncthreads();
            return;
        }

        batchCopy(heapItems + targetIdx * nodeSize, sMergedItems1, nodeSize);

        if (threadIdx.x == 0) {
            changeStatus(&status[currentIdx / 2], INUSE, AVAIL);
            changeStatus(&status[currentIdx], INUSE, AVAIL);
        }
        __syncthreads();
    }

    __device__ bool deleteRoot(int *items, int &size) {
        
        // change Status of the buffer to INUSE
        if (threadIdx.x == 0) {
            changeStatus(&status[0], AVAIL, INUSE);
        }
        __syncthreads();

        int deleteOffset = 0;

        // if the heap is empty we return false 
        if (*currNumNodes == 0 && *partialBufferSize == 0) {
            if (threadIdx.x == 0) {
                changeStatus(&status[0], INUSE, AVAIL);
            }
            size = 0;
            __syncthreads();
            return false;
        }

        // if there is no elements in the nodes but there is in the buffer
        if (*currNumNodes == 0 && *partialBufferSize != 0) {
            size = *partialBufferSize;
            batchCopy(items + deleteOffset, heapItems, size, true, INT_MAX);

            if (threadIdx.x == 0) {
                *partialBufferSize = 0;
                changeStatus(&status[0], INUSE, AVAIL);
            }
            __syncthreads();
            return false;
        }

        // change Status of the root node to INUSE
        if (threadIdx.x == 0) {
            changeStatus(&status[1], AVAIL, INUSE);
        }
        __syncthreads();

        size = nodeSize;
        batchCopy(items + deleteOffset, heapItems + nodeSize, size);
        return true;
    }

    // deleteUpdate is used to update the heap
    // it will fill the empty root batch
    __device__ void deleteUpdate(int smemOffset) {

        extern __shared__ int s[];
        int *sMergedItems = (int *)&s[smemOffset];
        int *tmpIdx = (int *)&s[smemOffset];

        smemOffset += sizeof(int) * 3 * nodeSize / sizeof(int);
        // smemOffset += (512 + 1) * sizeof(int) + 2 * nodeSize * sizeof(int); // added for testing

        int *tmpType = (int *)&s[smemOffset - 1];

        if (threadIdx.x == 0) {
            *tmpIdx = atomicSub(currNumNodes, 1);
            // if no more batches in the heap
            if (*tmpIdx == 1) {
                changeStatus(&status[1], INUSE, AVAIL);
                changeStatus(&status[0], INUSE, AVAIL);
            }
        }
        __syncthreads();

        int lastIdx = *tmpIdx;
        __syncthreads();

        if (lastIdx == 1) return;

        if (threadIdx.x == 0) {
            while(1) {
                if (atomicCAS(&status[lastIdx], AVAIL, INUSE) == AVAIL) {
                    *tmpType = 0;
                    break;
                }
                if (atomicCAS(&status[lastIdx], TARGET, MARKED) == TARGET) {
                    *tmpType = 1;
                    break;
                }
            }
        }
        __syncthreads();

        if (*tmpType == 1) {
            // wait for insert worker
            if (threadIdx.x == 0) {
                while (atomicCAS(&status[lastIdx], TARGET, AVAIL) != TARGET) {}
            }
            __syncthreads();

            batchCopy(sMergedItems, heapItems + nodeSize, nodeSize);
        }
        else if (*tmpType == 0){

            batchCopy(sMergedItems, heapItems + lastIdx * nodeSize, nodeSize, true, INT_MAX);

            if (threadIdx.x == 0) {
                changeStatus(&status[lastIdx], INUSE, AVAIL);
            }
            __syncthreads();
        }

        /* start handling partial batch */
        batchCopy(sMergedItems + nodeSize, heapItems, nodeSize);

        imergePath(sMergedItems, sMergedItems + nodeSize, sMergedItems, heapItems, nodeSize, smemOffset);
        __syncthreads();

        if (threadIdx.x == 0) {
            changeStatus(&status[0], INUSE, AVAIL);
        }
        __syncthreads();
        /* end handling partial batch */

        int currentIdx = 1;
        while (1) {
            int leftIdx = currentIdx << 1;
            int rightIdx = leftIdx + 1;
            // Wait until status[] are not locked
            // After that if the status become unlocked, than child exists
            // If the status is not unlocked, than no valid child
            if (threadIdx.x == 0) {
                int leftStatus, rightStatus;
                leftStatus = atomicCAS(&status[leftIdx], AVAIL, INUSE);
                while (leftStatus == INUSE) {
                    leftStatus = atomicCAS(&status[leftIdx], AVAIL, INUSE);
                }
                if (leftStatus != AVAIL) {
                    *tmpType = 0;
                }
                else {
                    rightStatus = atomicCAS(&status[rightIdx], AVAIL, INUSE);
                    while (rightStatus == INUSE) {
                        rightStatus = atomicCAS(&status[rightIdx], AVAIL, INUSE);
                    }
                    if (rightStatus != AVAIL) {
                        *tmpType = 1;
                    }
                    else {
                        *tmpType = 2;
                    }
                }
            }
            __syncthreads();

            int deleteType = *tmpType;
            __syncthreads();

            if (deleteType == 0) { // no children
                // move shared memory to currentIdx
                batchCopy(heapItems + currentIdx * nodeSize, sMergedItems, nodeSize);
                if (threadIdx.x == 0) {
                    changeStatus(&status[currentIdx], INUSE, AVAIL);
                }
                __syncthreads();
                return;
            }
            else if (deleteType == 1) { // only has left child and left child is a leaf batch
                // move leftIdx to shared memory
                batchCopy(sMergedItems + nodeSize, heapItems + leftIdx * nodeSize, nodeSize);

                imergePath(sMergedItems, sMergedItems + nodeSize, heapItems + currentIdx * nodeSize, heapItems + leftIdx * nodeSize, nodeSize, smemOffset);
                __syncthreads();

                if (threadIdx.x == 0) {
                    // unlock batch[currentIdx] & batch[leftIdx]
                    changeStatus(&status[currentIdx], INUSE, AVAIL);
                    changeStatus(&status[leftIdx], INUSE, AVAIL);
                }
                __syncthreads();
                return;
            }

            // move leftIdx and rightIdx to shared memory
            batchCopy(sMergedItems + nodeSize, heapItems + leftIdx * nodeSize, nodeSize);
            batchCopy(sMergedItems + 2 * nodeSize, heapItems + rightIdx * nodeSize, nodeSize);

            int largerIdx = (heapItems[leftIdx * nodeSize + nodeSize - 1] < heapItems[rightIdx * nodeSize + nodeSize - 1]) ? rightIdx : leftIdx;
            int smallerIdx = 4 * currentIdx - largerIdx + 1;
            __syncthreads();

            imergePath(sMergedItems + nodeSize, sMergedItems + 2 * nodeSize, sMergedItems + nodeSize, heapItems + largerIdx * nodeSize, nodeSize, smemOffset);
            __syncthreads();

            if (threadIdx.x == 0) {
                changeStatus(&status[largerIdx], INUSE, AVAIL);
            }
            __syncthreads();

            if (sMergedItems[0] >= sMergedItems[2 * nodeSize - 1]) {
                batchCopy(heapItems + currentIdx * nodeSize, sMergedItems + nodeSize, nodeSize);
            }
            else if (sMergedItems[nodeSize - 1] <= sMergedItems[nodeSize]) {
                batchCopy(heapItems + currentIdx * nodeSize, sMergedItems, nodeSize);
                batchCopy(heapItems + smallerIdx * nodeSize, sMergedItems + nodeSize, nodeSize);
                if (threadIdx.x == 0) {
                    changeStatus(&status[currentIdx], INUSE, AVAIL);
                    changeStatus(&status[smallerIdx], INUSE, AVAIL);
                }
                __syncthreads();
                return;
            }
            else {
                imergePath(sMergedItems, sMergedItems + nodeSize, heapItems + currentIdx * nodeSize, sMergedItems, nodeSize, smemOffset);
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                changeStatus(&status[currentIdx], INUSE, AVAIL);
            }
            __syncthreads();
            currentIdx = smallerIdx;
        }
    }
};
#endif

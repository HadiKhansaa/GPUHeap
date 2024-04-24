#ifndef MIN_HEAP_H
#define MIN_HEAP

#include <iostream>
#include <vector>
#include <limits>
#include "timer.h"

template<typename T>
class MinHeap {
private:
    std::vector<T> heap;

    int parent(int i) {
        return (i - 1) / 2;
    }

    int leftChild(int i) {
        return 2 * i + 1;
    }

    int rightChild(int i) {
        return 2 * i + 2;
    }

    void percolateDown(int i) {
        int size = heap.size();
        while (true) {
            int left = leftChild(i);
            int right = rightChild(i);
            int smallest = i;

            if (left < size && heap[left] < heap[smallest])
                smallest = left;
            if (right < size && heap[right] < heap[smallest])
                smallest = right;

            if (smallest != i) {
                std::swap(heap[i], heap[smallest]);
                i = smallest;
            } else {
                break;
            }
        }
    }

public:
    bool isEmpty() {
        return heap.empty();
    }

    size_t size() {
        return heap.size();
    }

    void buildHeap(std::vector<T> arr) {
        heap = arr;
        for (int i = (arr.size() / 2) - 1; i >= 0; --i)
            percolateDown(i);
    }

    void insertNode(T key) {
        heap.push_back(key);
        int current = size() - 1;
        while (current > 0 && heap[parent(current)] > heap[current]) {
            std::swap(heap[current], heap[parent(current)]);
            current = parent(current);
        }
    }

    T extractMin() {
        if (isEmpty())
            throw std::out_of_range("Cannot extract from an empty heap.");

        T root = heap[0];
        heap[0] = heap.back();
        heap.pop_back();
        percolateDown(0);
        return root;
    }

    void decreaseKey(int i, T new_val) {
        if (i < 0 || i >= size())
            throw std::out_of_range("Index is out of bounds.");
        if (new_val > heap[i])
            throw std::invalid_argument("New value is greater than the current value.");
        heap[i] = new_val;
        while (i != 0 && heap[parent(i)] > heap[i]) {
            std::swap(heap[i], heap[parent(i)]);
            i = parent(i);
        }
    }

    void increaseKey(int i, T new_val) {
        if (i < 0 || i >= size())
            throw std::out_of_range("Index is out of bounds.");
        if (new_val < heap[i])
            throw std::invalid_argument("New value is less than the current value. Use decreaseKey instead.");
        heap[i] = new_val;
        percolateDown(i);
    }

    T getMin() {
        if (isEmpty())
            return T(); // or any other default value
        return heap[0];
    }

    void deleteNode(int i) {
        if (i < 0 || i >= size())
            throw std::out_of_range("Index is out of bounds.");
        decreaseKey(i, std::numeric_limits<T>::min());
        extractMin();
    }

    std::vector<T> heapSort() {
        std::vector<T> sorted;
        while (!isEmpty()) {
            sorted.push_back(extractMin());
        }
        return sorted;
    }

    void replaceMin(T key) {
        if (isEmpty())
            throw std::out_of_range("Cannot replace the minimum of an empty heap.");
        heap[0] = key;
        percolateDown(0);
    }

    void merge(MinHeap<T>& otherHeap) {
        heap.insert(heap.end(), otherHeap.heap.begin(), otherHeap.heap.end());
        buildHeap(heap);
    }

    bool validateHeap() {
        for (size_t i = 0; i <= (size() - 2) / 2; ++i) {
            size_t left = leftChild(i);
            size_t right = rightChild(i);
            if (left < size() && heap[i] > heap[left])
                return false;
            if (right < size() && heap[i] > heap[right])
                return false;
        }
        return true;
    }
};

// Function to perform insertions and deletions and calculate time
void performOperations() {
    Timer timer;
    startTime(&timer); // Start the timer

    const int NUM_OPERATIONS = 100000000; // Define the number of operations

    MinHeap<int> minHeap;

    // Perform insertions
    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        minHeap.insertNode(rand() % 1000); // Insert random integers
    }

    // Perform deletions
    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        minHeap.extractMin();
    }

    stopTime(&timer); // Stop the timer

    // Print elapsed time
    printElapsedTime(timer, "Total time taken");
}

#endif

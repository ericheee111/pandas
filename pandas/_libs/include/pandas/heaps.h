/*
Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.

Dual-heap (max-heap + min-heap) data structure for O(log n) rolling
median computation with lazy deletion via an out_ind watermark.
*/
#pragma once

#include <stdlib.h>

typedef struct {
    double first;
    int second;
} pair;

static inline pair make_pair(double first, int second) {
    pair p;
    p.first = first;
    p.second = second;
    return p;
}

static inline int greater_pair(pair x, pair y) {
    if (x.first > y.first) return 1;
    if (x.first < y.first) return 0;
    return x.second > y.second;
}

static inline int less_pair(pair x, pair y) {
    if (x.first < y.first) return 1;
    if (x.first > y.first) return 0;
    return x.second < y.second;
}

typedef struct MaxHeap {
    pair* array;
    int size;
} MaxHeap;

static inline MaxHeap *maxheap_init(int max_size) {
    MaxHeap *heap = (MaxHeap *)malloc(sizeof(MaxHeap));
    if (heap == NULL) {
        return NULL;
    }
    max_size++;
    if (max_size <= 0) {
        free(heap);
        return NULL;
    }
    heap->array = (pair *)malloc((size_t)max_size * sizeof(pair));
    if (heap->array == NULL) {
        free(heap);
        return NULL;
    }
    heap->size = 0;
    return heap;
}

static inline void maxheap_destroy(MaxHeap *heap) {
    if (heap == NULL) {
        return;
    }
    free(heap->array);
    free(heap);
}

static inline void maxheap_heapify(MaxHeap *heap, int i) {
    pair value = heap->array[i];
    int n = heap->size;

    while ((i << 1) <= n) {
        int child = i << 1;
        int right = child | 1;

        if (right <= n && greater_pair(heap->array[right], heap->array[child])) {
            child = right;
        }

        if (!greater_pair(heap->array[child], value)) {
            break;
        }

        heap->array[i] = heap->array[child];
        i = child;
    }

    heap->array[i] = value;
}

static inline void maxheap_push(MaxHeap *heap, pair value) {
    int i = ++heap->size;
    while (i > 1 && greater_pair(value, heap->array[i >> 1])) {
        heap->array[i] = heap->array[i >> 1];
        i >>= 1;
    }
    heap->array[i] = value;
}

static inline void maxheap_pop(MaxHeap *heap) {
    if (heap->size <= 0) {
        return;
    }
    if (heap->size == 1) {
        heap->size--;
        return;
    }
    heap->array[1] = heap->array[heap->size];
    heap->size--;
    maxheap_heapify(heap, 1);
}

static inline pair maxheap_top(MaxHeap *heap) {
    return heap->array[1];
}

static inline int maxheap_size(MaxHeap *heap) {
    return heap->size;
}

typedef struct MinHeap {
    pair* array;
    int size;
} MinHeap;

static inline MinHeap *minheap_init(int max_size) {
    MinHeap *heap = (MinHeap *)malloc(sizeof(MinHeap));
    if (heap == NULL) {
        return NULL;
    }
    max_size++;
    if (max_size <= 0) {
        free(heap);
        return NULL;
    }
    heap->array = (pair *)malloc((size_t)max_size * sizeof(pair));
    if (heap->array == NULL) {
        free(heap);
        return NULL;
    }
    heap->size = 0;
    return heap;
}

static inline void minheap_destroy(MinHeap *heap) {
    if (heap == NULL) {
        return;
    }
    free(heap->array);
    free(heap);
}

static inline void minheap_heapify(MinHeap *heap, int i) {
    pair value = heap->array[i];
    int n = heap->size;

    while ((i << 1) <= n) {
        int child = i << 1;
        int right = child | 1;

        if (right <= n && less_pair(heap->array[right], heap->array[child])) {
            child = right;
        }

        if (!less_pair(heap->array[child], value)) {
            break;
        }

        heap->array[i] = heap->array[child];
        i = child;
    }

    heap->array[i] = value;
}

static inline void minheap_push(MinHeap *heap, pair value) {
    int i = ++heap->size;
    while (i > 1 && less_pair(value, heap->array[i >> 1])) {
        heap->array[i] = heap->array[i >> 1];
        i >>= 1;
    }
    heap->array[i] = value;
}

static inline void minheap_pop(MinHeap *heap) {
    if (heap->size <= 0) {
        return;
    }
    if (heap->size == 1) {
        heap->size--;
        return;
    }
    heap->array[1] = heap->array[heap->size];
    heap->size--;
    minheap_heapify(heap, 1);
}

static inline pair minheap_top(MinHeap *heap) {
    return heap->array[1];
}

static inline int minheap_size(MinHeap *heap) {
    return heap->size;
}

typedef struct heaps {
    MaxHeap *L;
    MinHeap *R;
    int cnt;
    int size;
    int out_ind;
} heaps;

static inline heaps* heaps_init(int n) {
    heaps *res = (heaps *)malloc(sizeof(heaps));
    if (res == NULL) {
        return NULL;
    }
    res->L = maxheap_init(n);
    res->R = minheap_init(n);
    if (res->L == NULL || res->R == NULL) {
        maxheap_destroy(res->L);
        minheap_destroy(res->R);
        free(res);
        return NULL;
    }
    res->cnt = 0;
    res->size = 0;
    res->out_ind = -1;
    return res;
}

static inline void heaps_insert(heaps *h, double value, int idx) {
    pair x = make_pair(value, idx);
    if (!maxheap_size(h->L) || !greater_pair(x, maxheap_top(h->L))) {
        maxheap_push(h->L, x);
    } else {
        minheap_push(h->R, x);
    }
    h->size++;
}

static inline double heaps_get(heaps* h, int k, int *ret) {
    if (k < 0 || k >= h->size) {
        *ret = 0;
        return 0;
    }
    ++k;
    while (maxheap_size(h->L) - k - h->cnt < 0) {
        if (!minheap_size(h->R)) {
            *ret = 0;
            return 0;
        }
        if (minheap_top(h->R).second > h->out_ind) {
            maxheap_push(h->L, minheap_top(h->R));
        }
        minheap_pop(h->R);
    }

    while (maxheap_size(h->L) - k - h->cnt > 0) {
        if (!maxheap_size(h->L)) {
            *ret = 0;
            return 0;
        }
        if (maxheap_top(h->L).second <= h->out_ind) {
            h->cnt--;
        } else {
            minheap_push(h->R, maxheap_top(h->L));
        }
        maxheap_pop(h->L);
    }
    while (maxheap_size(h->L) && maxheap_top(h->L).second <= h->out_ind) {
        h->cnt--;
        maxheap_pop(h->L);
    }
    if (!maxheap_size(h->L)) {
        *ret = 0;
        return 0;
    }
    *ret = 1;
    return maxheap_top(h->L).first;
}

static inline int heaps_get_adjacent(
    heaps *h,
    int k,
    double *x,
    double *y,
    int *ret
) {
    int local_ret = 0;

    if (k < 0 || k + 1 >= h->size) {
        *ret = 0;
        return 0;
    }

    *x = heaps_get(h, k, &local_ret);
    if (local_ret == 0) {
        *ret = 0;
        return 0;
    }

    while (minheap_size(h->R) && minheap_top(h->R).second <= h->out_ind) {
        minheap_pop(h->R);
    }
    if (minheap_size(h->R) == 0) {
        *ret = 0;
        return 0;
    }

    *y = minheap_top(h->R).first;

    *ret = 1;
    return 1;
}

static inline void heaps_remove(heaps* h, double value, int idx) {
    while (maxheap_size(h->L) && maxheap_top(h->L).second <= h->out_ind) {
        h->cnt--;
        maxheap_pop(h->L);
    }
    if (maxheap_size(h->L)) {
        h->cnt += !greater_pair(make_pair(value, idx), maxheap_top(h->L));
    }
    h->out_ind = idx;
    h->size--;
}

static inline void heaps_destroy(heaps *h) {
    maxheap_destroy(h->L);
    minheap_destroy(h->R);
    free(h);
}

static inline void maxheap_clear(MaxHeap *heap) {
    heap->size = 0;
}

static inline void minheap_clear(MinHeap *heap) {
    heap->size = 0;
}

static inline void heaps_clear(heaps *h) {
    maxheap_clear(h->L);
    minheap_clear(h->R);

    h->cnt = 0;
    h->size = 0;
    h->out_ind = -1;
}

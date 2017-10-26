/* C utilities related to factor graph computations */

#include "Python.h"
#include "numpy/arrayobject.h"
#include "dilation.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <vector>
#include <iostream>

using namespace std;


/* #### Globals #################################### */

/* ==== Set up the methods table ====================== */
static PyMethodDef dilationmethods[] = {
    {"max_filter1d", py_max_filter1d, METH_VARARGS},
    {"brute_max_filter1d", py_brute_max_filter1d, METH_VARARGS},
    {NULL, NULL}   /* Sentinel - marks the end of this structure */
};


/* ==== Initialize the C_test functions ====================== */
extern "C" {
void init_dilation()
{
    (void) Py_InitModule("_dilation", dilationmethods);
    import_array(); // Must be present for NumPy.  Called first after above line.
}
}

/// Check condition and return NULL (which will cause a python exception) if
/// it's false, and include an arbitrary format string as the error message
#define ASSERT(cond, ...) \
    if (!(cond)) {                          \
        char __ASSERTION_ERROR_MESSAGE__[1000]; \
        snprintf(__ASSERTION_ERROR_MESSAGE__, 1000, __VA_ARGS__); \
        PyErr_SetString(PyExc_AssertionError, __ASSERTION_ERROR_MESSAGE__); \
        return NULL; \
    }

void brute_max_filter1d(const float *cinMat, float *coutMat,
                        const int num_rows, const int num_cols, const int size, const int axis) {

    int lead = size / 2;
    const float *offset_in;
    float *cur_cout = coutMat;

    // Max across columns
    if (axis == 1)
    {
        // We can do axis=1 in place if we keep a buffer.
        // This means the wrapping functions don't need to build
        // larger scratch buffers.
        float buffer[num_cols];
        for (int r=0; r < num_rows; r++)
        {
            int offset = r*num_cols;
            offset_in = cinMat + offset;

            // initialize the starting point : best_index will have
            // the highest index from 0..size-lead, and best_score
            // will be the score
            int best_index = 0;
            float best_score = *offset_in;
            int end = std::min(size - lead, num_cols);
            for (int c = 1; c < end; ++c) {
                if (*(offset_in + c) > best_score) {
                    best_index = c;
                    best_score = *(offset_in + c);
                }
            }
            buffer[0] = best_score;

            // Now, for each index, we are in one of two regimes.
            // If the previous best has not fallen out of the window,
            // we only need to compare it to the next best.
            // Alternatively, if the previous best has fallen out of
            // the window, we just rebuild the whole window.
            for (int c = 1; c < num_cols; c++)
            {
                if (c - lead <= best_index) {
                    // best_index is still in the window
                    // only need to check the one new index
                    int pos = c + size - lead - 1;
                    if (pos < num_cols) {
                        if (*(offset_in + pos) >= best_score) {
                            best_score = *(offset_in + pos);
                            best_index = pos;
                        }
                    }
                } else {
                    int start = c - lead;
                    int end = std::min(c - lead + size, num_cols);
                    best_score = *(offset_in + start);
                    best_index = start;
                    for (int pos = start + 1; pos < end; ++pos) {
                        if (*(offset_in + pos) >= best_score) {
                            best_score = *(offset_in + pos);
                            best_index = pos;
                        }
                    }
                }
                buffer[c] = best_score;
            }
            for (int c = 0; c < num_cols; ++c)
            {
                *cur_cout++ = buffer[c];
            }
        }

    }
    else // Max across rows
    {
        for (int r = 0; r < num_rows; ++r)
        {
            int start = std::max(r - lead, 0);
            int end = std::min(r - lead + size, num_rows);
            const float *col_0_cin = cinMat + start * num_cols;
            for (int c = 0; c < num_cols; ++c)
            {
                const float *cur_cin = col_0_cin + c;
                float mx = *cur_cin;
                for (int pos = start + 1; pos < end; ++pos)
                {
                    cur_cin += num_cols;
                    if (*cur_cin > mx)
                    {
                        mx = *cur_cin;
                    }
                }
                *cur_cout++ = mx;
            }
        }
    }

}

/******************************************************************************
* max_filter1d
* Returns an ND array that is formed by taking max over a 1D sliding window.
* Currently only compatible with 2D arrays.
* When the output buffer is the same as the input, axis must be 0.
* ****************************************************************************/

static PyObject *py_brute_max_filter1d(PyObject *self, PyObject *args)
{
    PyArrayObject *inMat, *outMat; // The python objects to be extracted from the args
    int k, axis;

    /* Parse tuples separately since args will differ between C fcns */
    if (!PyArg_ParseTuple(args, "O!O!ii",
                          &PyArray_Type, &inMat,
                          &PyArray_Type, &outMat,
                          &k,
                          &axis
                          ))
        return NULL;
    if (NULL == inMat)
        return NULL;
    if (NULL == outMat)
        return NULL;

    float *cinMat = (float *) inMat->data;
    float *coutMat = (float *) outMat->data;

    int num_cols = inMat->dimensions[1];
    int num_rows = inMat->dimensions[0];

  #if DEBUG
    int target_flags = NPY_ALIGNED | NPY_C_CONTIGUOUS | NPY_OWNDATA;
    ASSERT( (PyArray_FLAGS(inMat) & target_flags) == target_flags, "Input array must be C contiguous.");
    ASSERT( (PyArray_FLAGS(outMat) & target_flags) == target_flags, "Input array must be C contiguous.");

    ASSERT(PyArray_NDIM(inMat) == 2, "Input array must be 2D");
    ASSERT(PyArray_NDIM(outMat) == 2, "Output array must be 2D");

    PyArray_Descr* desc = inMat->descr;
    ASSERT(desc->kind == 'f' && desc->elsize == sizeof(float),
           "Incorrect floating point type");

    desc = outMat->descr;
    ASSERT(desc->kind == 'f' && desc->elsize == sizeof(float),
           "Incorrect floating point type");

    ASSERT(num_rows == outMat->dimensions[0] && \
           num_cols == outMat->dimensions[1], "Input and output buffers must have the same shape.");
  #endif

    ASSERT(k > 1, "Cannot perform 1D maximum unless k>1.");

    brute_max_filter1d(cinMat, coutMat, num_rows, num_cols, k, axis);

    return Py_BuildValue("i", 1);
}

struct spair {
    float value;
    int death;
};

void max_filter1d(const float *cinMat, float *coutMat,
                  const int num_rows, const int num_cols, const int size, const int axis) {

    const float* row;
    float *out_row;
    int k = size;

    if (k > 2*num_rows-1 && axis==0)
        k = 2*num_rows-1;

    if (k > 2*num_cols-1 && axis==1)
        k = 2*num_cols-1;

    int hk = (k - 1) / 2;
    int half_k = k / 2;

    int max_queue = axis==1 ? num_cols : num_rows;
    struct spair pairs[max_queue+2];
    struct spair * maxpair;
    struct spair * last;
    int out_idx;
    int hk1 = half_k + 1;

    // Max across columns
    if (axis == 1)
    {
        for (int r=0; r < num_rows; r++)
        {
            int offset = r*num_cols;
            row = cinMat + offset;
            out_row = coutMat + offset;

            maxpair = pairs;
            last = pairs;
            maxpair->value = *row;
            maxpair->death = hk1;

            for (int c = 1; c < hk; c++)
            {
                ++row;
                if (*row >= maxpair->value) {
                    maxpair->value = *row;
                    maxpair->death = c + hk1;
                    last = maxpair;
                } else {
                    while (last->value <= *row) {
                        --last;
                    }
                    ++last;
                    last->value = *row;
                    last->death = c + hk1;
                }
            }

            for (int c = hk; c < num_cols; c++)
            {
                ++row;
                if (maxpair->death == c-hk)
                    maxpair++;

                if (*row >= maxpair->value) {
                    maxpair->value = *row;
                    maxpair->death = c + hk1;
                    last = maxpair;
                } else {
                    while (last->value <= *row) {
                        --last;
                    }
                    ++last;
                    last->value = *row;
                    last->death = c + hk1;
                }
                out_row[c - hk] = maxpair->value;
            }

            for (int c = num_cols-hk; c < num_cols; c++)
            {
                if (maxpair->death == c)
                    maxpair++;

                out_row[c] = maxpair->value;
            }
        }
        // Max across rows
    }
    else
    {
        float line_buffer[num_rows];
        float * col;

        for (int c = 0; c < num_cols; c++)
        {
            // Copy the column before iterating. The indexing is faster this way.
            out_idx = c;
            for (int r = 0; r < num_rows; r++)
            {
                line_buffer[r] = cinMat[out_idx];
                out_idx += num_cols;
            }

            col = line_buffer;

            maxpair = pairs;
            last = pairs;
            maxpair->value = *col;
            maxpair->death = hk1;

            for (int r = 1; r < hk; r++)
            {
                ++col;
                if (*col >= maxpair->value) {
                    maxpair->value = *col;
                    maxpair->death = r + hk1;
                    last = maxpair;
                } else {
                    while (last->value <= *col) {
                        --last;
                    }
                    ++last;
                    last->value = *col;
                    last->death = r + hk1;
                }
            }

            out_idx = c;
            for (int r = hk; r < num_rows; r++)
            {
                ++col;
                if (maxpair->death == r-hk)
                    maxpair++;

                if (*col >= maxpair->value) {
                    maxpair->value = *col;
                    maxpair->death = r + hk1;
                    last = maxpair;
                } else {
                    while (last->value <= *col) {
                        --last;
                    }
                    ++last;
                    last->value = *col;
                    last->death = r + hk1;
                }

                coutMat[out_idx] = maxpair->value;
                out_idx += num_cols;
            }

            for (int r = num_rows-hk; r < num_rows; r++)
            {
                if (maxpair->death == r)
                    maxpair++;

                coutMat[out_idx] = maxpair->value;
                out_idx += num_cols;
            }
        }
    }

}


static PyObject *py_max_filter1d(PyObject *self, PyObject *args)
{
    PyArrayObject *inMat, *outMat; // The python objects to be extracted from the args
    int k, axis;

    /* Parse tuples separately since args will differ between C fcns */
    if (!PyArg_ParseTuple(args, "O!O!ii",
                          &PyArray_Type, &inMat,
                          &PyArray_Type, &outMat,
                          &k,
                          &axis
                          ))
        return NULL;
    if (NULL == inMat)
        return NULL;
    if (NULL == outMat)
        return NULL;

    float *cinMat = (float *) inMat->data;
    float *coutMat = (float *) outMat->data;

    int num_rows = inMat->dimensions[0];
    int num_cols = inMat->dimensions[1];

  #if DEBUG
    int target_flags = NPY_ALIGNED | NPY_C_CONTIGUOUS | NPY_OWNDATA;
    ASSERT( (PyArray_FLAGS(inMat) & target_flags) == target_flags, "Input array must be C contiguous.");
    ASSERT( (PyArray_FLAGS(outMat) & target_flags) == target_flags, "Input array must be C contiguous.");

    ASSERT(PyArray_NDIM(inMat) == 2, "Input array must be 2D");
    ASSERT(PyArray_NDIM(outMat) == 2, "Output array must be 2D");

    PyArray_Descr* desc = inMat->descr;
    ASSERT(desc->kind == 'f' && desc->elsize == sizeof(float),
           "Incorrect floating point type");

    desc = outMat->descr;
    ASSERT(desc->kind == 'f' && desc->elsize == sizeof(float),
           "Incorrect floating point type");

    ASSERT(num_rows == outMat->dimensions[0] && \
           num_cols == outMat->dimensions[1], "Input and output buffers must have the same shape.");
  #endif

    max_filter1d(cinMat, coutMat, num_rows, num_cols, k, axis);

    return Py_BuildValue("i", 1);
}

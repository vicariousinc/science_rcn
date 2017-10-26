/* Header for dilation.cc */

/* ==== Prototypes =================================== */
void brute_max_filter1d(const float *cinMat, float *coutMat, 
                        const int num_rows, const int num_cols, const int size, const int axis);
void max_filter1d(const float *cinMat, float *coutMat, 
                  const int num_rows, const int num_cols, const int size, const int axis);

static PyObject *py_brute_max_filter1d(PyObject *self, PyObject *args);
static PyObject *py_max_filter1d(PyObject *self, PyObject *args);
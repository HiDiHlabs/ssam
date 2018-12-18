#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_max_threads() { return 1;}
#endif

#include <Python.h>
#include "numpy/npy_math.h"
#include "numpy/arrayobject.h"

#define I2D(X, Y, YL) ((X) * (YL) + (Y))
#define I3D(X, Y, Z, YL, ZL) (((X) * (YL) * (ZL)) + ((Y) * (ZL)) + (Z))

static double __corr__(double *a, double *b, int ngene) {
    double a_mean = 0;
    double b_mean = 0;
    double aa_mean = 0;
    double bb_mean = 0;
    double a_std = 0;
    double b_std = 0;
    double rtn = 0;
    int i;

    for (i=0; i<ngene; i++) {
        a_mean += a[i];
        b_mean += b[i];
        aa_mean += a[i]*a[i];
        bb_mean += b[i]*b[i];
    }

    a_mean /= ngene;
    b_mean /= ngene;
    aa_mean /= ngene;
    bb_mean /= ngene;

    a_std = sqrt(aa_mean - a_mean * a_mean);
    b_std = sqrt(bb_mean - b_mean * b_mean);

    if (a_std == 0 || b_std == 0) {
        rtn = 0;
    } else {
        for (i=0; i<ngene; i++)
            rtn += (a[i] - a_mean) * (b[i] - b_mean);
        rtn /= a_std * b_std;
    }

    rtn /= ngene;
    return rtn;
}

static PyObject *calc_corrmap(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg1 = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *oarr = NULL;
    long x, y, z, nd, ngene = 0;
    double *vecs, *corrmap;
    npy_intp *dimsp;
    int ncores = omp_get_max_threads();

    static char *kwlist[] = { "vf", "ncores", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", kwlist, &arg1, &ncores)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;   
    nd = PyArray_NDIM(arr1);
    if (nd != 3 && nd != 4) goto fail; // only 2D or 3D array is expected
    dimsp = PyArray_DIMS(arr1);
    oarr = (PyArrayObject*)PyArray_ZEROS(nd - 1, dimsp, NPY_DOUBLE, NPY_CORDER);
    ngene = dimsp[nd-1];
    corrmap = (double *)PyArray_DATA(oarr);
    vecs = (double *)PyArray_DATA(arr1);

    if (nd == 3) {
        // 2D
        #pragma omp parallel for collapse(2) num_threads(ncores)
        for (x=1; x<dimsp[0]-1; x++) {
            for (y=1; y<dimsp[1]-1; y++) {
                corrmap[I2D(x, y, dimsp[1])] =
                    __corr__(vecs + I2D(x, y, dimsp[1])*ngene, vecs + I2D(x+1, y, dimsp[1])*ngene, ngene) +
                    __corr__(vecs + I2D(x, y, dimsp[1])*ngene, vecs + I2D(x+1, y+1, dimsp[1])*ngene, ngene) +
                    __corr__(vecs + I2D(x, y, dimsp[1])*ngene, vecs + I2D(x+1, y-1, dimsp[1])*ngene, ngene) +
                    __corr__(vecs + I2D(x, y, dimsp[1])*ngene, vecs + I2D(x-1, y, dimsp[1])*ngene, ngene) +
                    __corr__(vecs + I2D(x, y, dimsp[1])*ngene, vecs + I2D(x-1, y+1, dimsp[1])*ngene, ngene) +
                    __corr__(vecs + I2D(x, y, dimsp[1])*ngene, vecs + I2D(x-1, y-1, dimsp[1])*ngene, ngene) +
                    __corr__(vecs + I2D(x, y, dimsp[1])*ngene, vecs + I2D(x, y+1, dimsp[1])*ngene, ngene) +
                    __corr__(vecs + I2D(x, y, dimsp[1])*ngene, vecs + I2D(x, y-1, dimsp[1])*ngene, ngene);
                corrmap[I2D(x, y, dimsp[1])] /= 8;
            }
        }
        #pragma omp parallel for num_threads(ncores)
        for (x=0; x<dimsp[0]; x++) {
            corrmap[I2D(x, 0, dimsp[1])] = NPY_NAN;
            corrmap[I2D(x, dimsp[1]-1, dimsp[1])] = NPY_NAN;
        }
        #pragma omp parallel for num_threads(ncores)
        for (y=0; y<dimsp[1]; y++) {
            corrmap[I2D(0, y, dimsp[1])] = NPY_NAN;
            corrmap[I2D(dimsp[0]-1, y, dimsp[1])] = NPY_NAN;
        }
    } else {
        // 3D
        #pragma omp parallel for collapse(3) num_threads(ncores)
        for (x=1; x<dimsp[0]-1; x++) {
            for (y=1; y<dimsp[1]-1; y++) {
                for (z=1; z<dimsp[2]-1; z++) {
                    corrmap[I3D(x, y, z, dimsp[1], dimsp[2])] =
                        __corr__(vecs + I3D(x,   y,   z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x+1, y,   z  , dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x,   y,   z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x+1, y+1, z  , dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x,   y,   z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x+1, y-1, z  , dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x,   y,   z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x+1, y,   z+1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x,   y,   z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x+1, y+1, z+1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x+1, y-1, z+1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x+1, y  , z-1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x+1, y+1, z-1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x+1, y-1, z-1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x-1, y  , z  , dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x-1, y+1, z  , dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x-1, y-1, z  , dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x  , y+1, z  , dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x  , y-1, z  , dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x-1, y  , z-1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x-1, y+1, z-1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x-1, y-1, z-1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x  , y+1, z-1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x  , y-1, z-1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x-1, y  , z+1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x-1, y+1, z+1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x-1, y-1, z+1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x  , y+1, z+1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x  , y-1, z+1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x  , y  , z+1, dimsp[1], dimsp[2])*ngene, ngene) +
                        __corr__(vecs + I3D(x  , y  , z  , dimsp[1], dimsp[2])*ngene,
                                 vecs + I3D(x  , y  , z-1, dimsp[1], dimsp[2])*ngene, ngene);
                    corrmap[I3D(x, y, z, dimsp[1], dimsp[2])] /= 26;
                }
            }
        }
        #pragma omp parallel for num_threads(ncores)
        for (x=1; x<dimsp[0]-1; x++) {
            corrmap[I3D(x, 0         , 0         , dimsp[1], dimsp[2])] = NPY_NAN;
            corrmap[I3D(x, dimsp[1]-1, 0         , dimsp[1], dimsp[2])] = NPY_NAN;
            corrmap[I3D(x, 0         , dimsp[2]-1, dimsp[1], dimsp[2])] = NPY_NAN;
            corrmap[I3D(x, dimsp[1]-1, dimsp[2]-1, dimsp[1], dimsp[2])] = NPY_NAN;
        }
        #pragma omp parallel for num_threads(ncores)
        for (y=1; y<dimsp[1]-1; y++) {
            corrmap[I3D(0         , y, 0         , dimsp[1], dimsp[2])] = NPY_NAN;
            corrmap[I3D(dimsp[0]-1, y, 0         , dimsp[1], dimsp[2])] = NPY_NAN;
            corrmap[I3D(0         , y, dimsp[2]-1, dimsp[1], dimsp[2])] = NPY_NAN;
            corrmap[I3D(dimsp[0]-1, y, dimsp[2]-1, dimsp[1], dimsp[2])] = NPY_NAN;
        }
        #pragma omp parallel for num_threads(ncores)
        for (z=1; z<dimsp[2]-1; z++) {
            corrmap[I3D(0         , 0         , z, dimsp[1], dimsp[2])] = NPY_NAN;
            corrmap[I3D(0         , dimsp[1]-1, z, dimsp[1], dimsp[2])] = NPY_NAN;
            corrmap[I3D(dimsp[0]-1, 0         , z, dimsp[1], dimsp[2])] = NPY_NAN;
            corrmap[I3D(dimsp[0]-1, dimsp[1]-1, z, dimsp[1], dimsp[2])] = NPY_NAN;
        }
        #pragma omp parallel for collapse(2) num_threads(ncores)
        for (x=1; x<dimsp[0]-1; x++) {
            for (y=1; y<dimsp[1]-1; y++) {
                corrmap[I3D(x, y, 0         , dimsp[1], dimsp[2])] = NPY_NAN;
                corrmap[I3D(x, y, dimsp[2]-1, dimsp[1], dimsp[2])] = NPY_NAN;
            }
        }
        #pragma omp parallel for collapse(2) num_threads(ncores)
        for (x=1; x<dimsp[0]-1; x++) {
            for (z=1; z<dimsp[2]-1; z++) {
                corrmap[I3D(x, 0         , z, dimsp[1], dimsp[2])] = NPY_NAN;
                corrmap[I3D(x, dimsp[1]-1, z, dimsp[1], dimsp[2])] = NPY_NAN;
            }
        }
        #pragma omp parallel for collapse(2) num_threads(ncores)
        for (y=1; y<dimsp[1]-1; y++) {
            for (z=1; z<dimsp[2]-1; z++) {
                corrmap[I3D(0         , y, z, dimsp[1], dimsp[2])] = NPY_NAN;
                corrmap[I3D(dimsp[0]-1, y, z, dimsp[1], dimsp[2])] = NPY_NAN;
            }
        }
        corrmap[I3D(0         , 0         , 0         , dimsp[1], dimsp[2])] = NPY_NAN;
        corrmap[I3D(0         , 0         , dimsp[2]-1, dimsp[1], dimsp[2])] = NPY_NAN;
        corrmap[I3D(0         , dimsp[1]-1, 0         , dimsp[1], dimsp[2])] = NPY_NAN;
        corrmap[I3D(0         , dimsp[1]-1, dimsp[2]-1, dimsp[1], dimsp[2])] = NPY_NAN;
        corrmap[I3D(dimsp[0]-1, 0         , 0         , dimsp[1], dimsp[2])] = NPY_NAN;
        corrmap[I3D(dimsp[0]-1, 0         , dimsp[2]-1, dimsp[1], dimsp[2])] = NPY_NAN;
        corrmap[I3D(dimsp[0]-1, dimsp[1]-1, 0         , dimsp[1], dimsp[2])] = NPY_NAN;
        corrmap[I3D(dimsp[0]-1, dimsp[1]-1, dimsp[2]-1, dimsp[1], dimsp[2])] = NPY_NAN;
    }
    Py_DECREF(arr1);

    return (PyObject *) oarr;
 fail:
    Py_XDECREF(arr1);
    return NULL;
}


static PyObject *calc_ctmap(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *arr2 = NULL;
    PyArrayObject *oarr = NULL;
    long nvec, nd, ngene = 0;
    double *cent, *vecs, *scores;
    npy_intp *dimsp;
    int ncores = omp_get_max_threads();
    int i;

    static char *kwlist[] = { "vec", "vf", "ncores", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|i", kwlist, &arg1, &arg2, &ncores)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;
    if ((arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) goto fail;
    if (PyArray_NDIM(arr1) != 1) goto fail;
    nd = PyArray_NDIM(arr2);
    if((ngene = *PyArray_DIMS(arr1)) != PyArray_DIMS(arr2)[nd-1]) goto fail;

    dimsp = PyArray_DIMS(arr2);
    oarr = (PyArrayObject*)PyArray_ZEROS(nd - 1, dimsp, NPY_DOUBLE, NPY_CORDER);

    nvec = 1;
    for (i=0; i<nd-1; i++)
        nvec *= dimsp[i];

    scores = (double *)PyArray_DATA(oarr);
    cent = (double *)PyArray_DATA(arr1);
    vecs = (double *)PyArray_DATA(arr2);

    #pragma omp parallel for num_threads(ncores)
    for (i=0; i<nvec; i++) {
        scores[i] = __corr__(cent, vecs + (i*ngene), ngene);
    }

    Py_DECREF(arr1);
    Py_DECREF(arr2);

    return (PyObject *) oarr;
 fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    return NULL;
}

static PyObject *corr(PyObject *self, PyObject *args) {
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *arr2 = NULL;
    long ngene = 0;
    double *a;
    double *b;
    double rtn = 0;

    if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;
    if ((arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) goto fail;
    if (PyArray_NDIM(arr1) != 1) goto fail;
    if (PyArray_NDIM(arr2) != 1) goto fail;
    if((ngene = *PyArray_DIMS(arr1)) != *PyArray_DIMS(arr2)) goto fail;

    a = (double *)PyArray_DATA(arr1);
    b = (double *)PyArray_DATA(arr2);

    rtn = __corr__(a, b, ngene);

    Py_DECREF(arr1);
    Py_DECREF(arr2);

    return PyFloat_FromDouble(rtn);
 fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    return NULL;
}

static struct PyMethodDef module_methods[] = {
    {"corr", (PyCFunction)corr, METH_VARARGS, "Calculates Pearson's correlation coefficient."},
    {"calc_ctmap", (PyCFunction)calc_ctmap, METH_VARARGS | METH_KEYWORDS, "Creates a cell type map."},
    {"calc_corrmap", (PyCFunction)calc_corrmap, METH_VARARGS | METH_KEYWORDS, "Creates a correlation map."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "analysis_utils",
        NULL,
        -1,
        module_methods
};
#endif

PyMODINIT_FUNC
PyInit_utils(void)
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    Py_InitModule("utils", module_methods);
#endif
    import_array();
#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
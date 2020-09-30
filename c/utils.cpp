#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <math.h>
#include <queue>
#include <map>

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

struct pos2d {
    long x;
    long y;
};

struct pos3d {
    long x;
    long y;
    long z;
};
bool operator< (pos3d a, pos3d b) { return (a.x == b.x && a.y == b.y)?(a.z < b.z):((a.x == b.x)?(a.y < b.y):(a.x < b.x)); }


static double gauss_kernel(double x, double y, double z) {
    return exp(-0.5 * (x*x + y*y + z*z)); // this is not normalized
}

void kde(std::map<pos3d, double> &arr, double *xx, double *yy, double *zz, int *shape, int npts, double bandwidth, double prune_coeff, int ncores) {
    int maxdist, xs, xe, ys, ye, zs, ze;
    double value;
    std::map<pos3d, double>::iterator it;
    if (prune_coeff > 0) {
        maxdist = (int)(bandwidth * prune_coeff);
    } else {
        maxdist = -1;
    }
    omp_lock_t lock;
    omp_init_lock(&lock);
    #pragma omp parallel for num_threads(ncores) private(it, value, xs, xe, ys, ye, zs, ze)
    for (int i=0; i<npts; i++) {
        int x0 = (int)xx[i];
        int y0 = (int)yy[i];
        int z0 = (int)zz[i];
        if (maxdist > 0) {
            xs = x0-maxdist;
            xe = x0+maxdist+1;
            ys = y0-maxdist;
            ye = y0+maxdist+1;
            zs = z0-maxdist;
            ze = z0+maxdist+1;
        } else {
            xs = 0;
            xe = shape[0];
            ys = 0;
            ye = shape[1];
            zs = 0;
            ze = shape[2];
        }
        for (int x=xs; x<xe; x++) {
            if (x < 0 || x >= shape[0])
                continue;
            for (int y=ys; y<ye; y++) {
                if (y < 0 || y >= shape[1])
                    continue;
                for (int z=zs; z<ze; z++) {
                    if (z < 0 || z >= shape[2])
                        continue;
                    pos3d key{.x = x, .y = y, .z = z};
                    value = gauss_kernel((x-xx[i])/bandwidth, (y-yy[i])/bandwidth, (z-zz[i])/bandwidth);
                    omp_set_lock(&lock);
                    it = arr.find(key);
                    if (it == arr.end())
                        arr[key] = value;
                    else
                        it->second += value;
                    omp_unset_lock(&lock);
                }
            }
        }
    }
    omp_destroy_lock(&lock);
}

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

static PyObject *calc_kde(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    PyObject *arg4 = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *arr2 = NULL;
    PyArrayObject *arr3 = NULL;
    PyArrayObject *arr4 = NULL;
    int ncores = omp_get_max_threads();
    PyObject *rtn, *poslist, *xlist, *ylist, *zlist, *vlist;
    double *x, *y, *z;
    int *shape;
    double h, prune_coeff;
    int kernel = 0;
    unsigned int npts;
    int cnt;
    std::map<pos3d, double> oarr_map;
    std::map<pos3d, double>::iterator it;

    static const char *kwlist[] = { "h", "x", "y", "z", "shape", "prune_coeff", "kernel", "ncores", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dOOOOd|ii", const_cast<char **>(kwlist), &h, &arg1, &arg2, &arg3, &arg4, &prune_coeff, &kernel, &ncores)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;
    if ((arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) goto fail;
    if ((arr3 = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) goto fail;
    if ((arr4 = (PyArrayObject*)PyArray_FROM_OTF(arg4, NPY_INT, NPY_ARRAY_IN_ARRAY)) == NULL) goto fail;
    
    if (PyArray_NDIM(arr1) != 1 || PyArray_NDIM(arr2) != 1 || PyArray_NDIM(arr3) != 1 || PyArray_NDIM(arr4) != 1)
    {
        goto fail;
    }

    npts = PyArray_DIMS(arr1)[0];

    x = (double *)PyArray_DATA(arr1);
    y = (double *)PyArray_DATA(arr2);
    z = (double *)PyArray_DATA(arr3);
    shape = (int *)PyArray_DATA(arr4);

    kde(oarr_map, x, y, z, shape, npts, h, prune_coeff, ncores);
    rtn = (PyObject *)PyTuple_New(2);
    poslist = (PyObject *)PyList_New(3);
    xlist = (PyObject *)PyList_New(oarr_map.size());
    ylist = (PyObject *)PyList_New(oarr_map.size());
    zlist = (PyObject *)PyList_New(oarr_map.size());
    vlist = (PyObject *)PyList_New(oarr_map.size());
    
    cnt = 0;
    for (it = oarr_map.begin(); it != oarr_map.end(); it++)
    {
        PyList_SetItem(xlist, cnt, PyLong_FromLong((long)(it->first.x)));
        PyList_SetItem(ylist, cnt, PyLong_FromLong((long)(it->first.y)));
        PyList_SetItem(zlist, cnt, PyLong_FromLong((long)(it->first.z)));
        PyList_SetItem(vlist, cnt, PyFloat_FromDouble(it->second));
        cnt++;
    }
    PyList_SetItem(poslist, 0, xlist);
    PyList_SetItem(poslist, 1, ylist);
    PyList_SetItem(poslist, 2, zlist);
    PyTuple_SetItem(rtn, 0, poslist);
    PyTuple_SetItem(rtn, 1, vlist);
    
    Py_DECREF(arr1);
    Py_DECREF(arr2);
    Py_DECREF(arr3);
    Py_DECREF(arr4);
    
    return (PyObject *) rtn;
    
fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    Py_XDECREF(arr3);
    Py_XDECREF(arr4);
    return NULL;
}

static PyObject *flood_fill(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject* filled_poslist = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *arr2 = NULL;
    long nvec, nd, ngene = 0;
    long *pos, x, y, z, cnt;
    double r = 0.6, *vf;
    npy_intp *dimsp;
    int min_pixels = 10, max_pixels=2000;
    int i;
    bool *mask;

    static const char *kwlist[] = { "pos", "vf", "r", "min_pixels", "max_pixels", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|dii", const_cast<char **>(kwlist), &arg1, &arg2, &r, &min_pixels, &max_pixels)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_LONG, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;
    if ((arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) goto fail;
    if (PyArray_NDIM(arr1) != 1) goto fail;
    nd = PyArray_NDIM(arr2);
    dimsp = PyArray_DIMS(arr2);
    nvec = 1;
    for (i=0; i<nd-1; i++)
        nvec *= dimsp[i];
    ngene = dimsp[nd-1];
    filled_poslist = PyList_New(0);
    pos = (long *)PyArray_DATA(arr1);
    vf = (double *)PyArray_DATA(arr2);
    mask = (bool *) calloc(nvec, sizeof(bool));
    cnt = 0;
    if (nd == 3) {
        // 2D
        std::queue<pos2d> queue2d;
        queue2d.push(pos2d());
        queue2d.back().x = pos[0];
        queue2d.back().y = pos[1];
        while (queue2d.size() > 0) {
            x = queue2d.front().x;
            y = queue2d.front().y;
            PyObject *t = PyTuple_New(2);
            PyTuple_SetItem(t, 0, PyLong_FromLong(x));
            PyTuple_SetItem(t, 1, PyLong_FromLong(y));
            cnt += 1;
            if (cnt > max_pixels) 
                break;
            PyList_Append(filled_poslist, t);
            queue2d.pop();
            if (x < dimsp[0] - 1 && mask[I2D(x + 1, y, dimsp[1])] == false &&
                    __corr__(vf + (I2D(pos[0], pos[1], dimsp[1]) * ngene),
                             vf + (I2D(x + 1, y, dimsp[1]) * ngene), ngene) > r) {
                mask[I2D(x + 1, y, dimsp[1])] = true;
                queue2d.push(pos2d());
                queue2d.back().x = x + 1;
                queue2d.back().y = y;
            }
            if (x > 1 && mask[I2D(x - 1, y, dimsp[1])] == false &&
                    __corr__(vf + (I2D(pos[0], pos[1], dimsp[1]) * ngene),
                             vf + (I2D(x - 1, y, dimsp[1]) * ngene), ngene) > r) {
                mask[I2D(x - 1, y, dimsp[1])] = true;
                queue2d.push(pos2d());
                queue2d.back().x = x - 1;
                queue2d.back().y = y;
            }
            if (y < dimsp[1] - 1 && mask[I2D(x, y + 1, dimsp[1])] == false &&
                    __corr__(vf + (I2D(pos[0], pos[1], dimsp[1]) * ngene),
                             vf + (I2D(x, y + 1, dimsp[1]) * ngene), ngene) > r) {
                mask[I2D(x, y + 1, dimsp[1])] = true;
                queue2d.push(pos2d());
                queue2d.back().x = x;
                queue2d.back().y = y + 1;
            }
            if (y > 1 && mask[I2D(x, y - 1, dimsp[1])] == false &&
                    __corr__(vf + (I2D(pos[0], pos[1], dimsp[1]) * ngene),
                             vf + (I2D(x, y - 1, dimsp[1]) * ngene), ngene) > r) {
                mask[I2D(x, y - 1, dimsp[1])] = true;
                queue2d.push(pos2d());
                queue2d.back().x = x;
                queue2d.back().y = y - 1;
            }
        }
    } else if (nd == 4) {
        // 3D
        std::queue<pos3d> queue3d;
        queue3d.push(pos3d());
        queue3d.back().x = pos[0];
        queue3d.back().y = pos[1];
        queue3d.back().z = pos[2];
        while (queue3d.size() > 0) {
            x = queue3d.front().x;
            y = queue3d.front().y;
            z = queue3d.front().z;
            PyObject *t = PyTuple_New(3);
            PyTuple_SetItem(t, 0, PyLong_FromLong(x));
            PyTuple_SetItem(t, 1, PyLong_FromLong(y));
            PyTuple_SetItem(t, 2, PyLong_FromLong(z));
            PyList_Append(filled_poslist, t);
            cnt += 1;
            if (cnt > max_pixels) 
                break;
            queue3d.pop();
            if (x < dimsp[0] - 1 && mask[I3D(x + 1, y, z, dimsp[1], dimsp[2])] == false &&
                    __corr__(vf + I3D(pos[0], pos[1], pos[2], dimsp[1], dimsp[2]) * ngene,
                             vf + I3D(x + 1, y, z, dimsp[1], dimsp[2]) * ngene, ngene) > r) {
                mask[I3D(x + 1, y, z, dimsp[1], dimsp[2])] = true;
                queue3d.push(pos3d());
                queue3d.back().x = x + 1;
                queue3d.back().y = y;
                queue3d.back().z = z;
            }
            if (x > 1 && mask[I3D(x - 1, y, z, dimsp[1], dimsp[2])] == false &&
                    __corr__(vf + I3D(pos[0], pos[1], pos[2], dimsp[1], dimsp[2]) * ngene,
                             vf + I3D(x - 1, y, z, dimsp[1], dimsp[2]) * ngene, ngene) > r) {
                mask[I3D(x - 1, y, z, dimsp[1], dimsp[2])] = true;
                queue3d.push(pos3d());
                queue3d.back().x = x - 1;
                queue3d.back().y = y;
                queue3d.back().z = z;
            }
            if (y < dimsp[1] - 1 && mask[I3D(x, y + 1, z, dimsp[1], dimsp[2])] == false &&
                    __corr__(vf + I3D(pos[0], pos[1], pos[2], dimsp[1], dimsp[2]) * ngene,
                             vf + I3D(x, y + 1, z, dimsp[1], dimsp[2]) * ngene, ngene) > r) {
                mask[I3D(x, y + 1, z, dimsp[1], dimsp[2])] = true;
                queue3d.push(pos3d());
                queue3d.back().x = x;
                queue3d.back().y = y + 1;
                queue3d.back().z = z;
            }
            if (y > 1 && mask[I3D(x, y - 1, z, dimsp[1], dimsp[2])] == false &&
                    __corr__(vf + I3D(pos[0], pos[1], pos[2], dimsp[1], dimsp[2]) * ngene,
                             vf + I3D(x, y - 1, z, dimsp[1], dimsp[2]) * ngene, ngene) > r) {
                mask[I3D(x, y - 1, z, dimsp[1], dimsp[2])] = true;
                queue3d.push(pos3d());
                queue3d.back().x = x;
                queue3d.back().y = y - 1;
                queue3d.back().z = z;
            }
            if (z < dimsp[2] - 1 && mask[I3D(x, y, z + 1, dimsp[1], dimsp[2])] == false &&
                    __corr__(vf + I3D(pos[0], pos[1], pos[2], dimsp[1], dimsp[2]) * ngene,
                             vf + I3D(x, y, z + 1, dimsp[1], dimsp[2]) * ngene, ngene) > r) {
                mask[I3D(x, y, z, dimsp[1], dimsp[2])] = true;
                queue3d.push(pos3d());
                queue3d.back().x = x;
                queue3d.back().y = y;
                queue3d.back().z = z + 1;
            }
            if (z > 1 && mask[I3D(x, y, z - 1, dimsp[1], dimsp[2])] == false &&
                    __corr__(vf + I3D(pos[0], pos[1], pos[2], dimsp[1], dimsp[2]) * ngene,
                             vf + I3D(x, y, z - 1, dimsp[1], dimsp[2]) * ngene, ngene) > r) {
                mask[I3D(x, y, z - 1, dimsp[1], dimsp[2])] = true;
                queue3d.push(pos3d());
                queue3d.back().x = x;
                queue3d.back().y = y;
                queue3d.back().z = z - 1;
            }
        }
    }
    free((void*)mask);
    Py_DECREF(arr1);
    Py_DECREF(arr2);
    if (cnt > max_pixels || cnt < min_pixels) 
        PyList_SetSlice(filled_poslist, 0, PyList_Size(filled_poslist), NULL);
    return (PyObject *) filled_poslist;
 
fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    return NULL;
}

static PyObject *calc_corrmap(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg1 = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *oarr = NULL;
    long i, x, y, z, dx, dy, dz;
    long nvec, nd, ngene = 0;
    double *vecs, *corrmap;
    npy_intp *dimsp;
    int ncores = omp_get_max_threads();
    int csize = 1;
    double *tmpvec;

    static const char *kwlist[] = { "vf", "ncores", "size", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ii", const_cast<char **>(kwlist), &arg1, &ncores, &csize)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;   
    nd = PyArray_NDIM(arr1);
    if (nd != 3 && nd != 4) goto fail; // only 2D or 3D array is expected
    dimsp = PyArray_DIMS(arr1);
    oarr = (PyArrayObject*)PyArray_ZEROS(nd - 1, dimsp, NPY_DOUBLE, NPY_CORDER);
    ngene = dimsp[nd-1];
    corrmap = (double *)PyArray_DATA(oarr);
    vecs = (double *)PyArray_DATA(arr1);
    nvec = 1;
    for (i=0; i<nd-1; i++)
        nvec *= dimsp[i];
    
    // initialize corrmap with NANs
    #pragma omp parallel for num_threads(ncores)
    for (i=0; i<nvec; i++)
        corrmap[i] = NPY_NAN;

    if (nd == 3) {
        // 2D
        #pragma omp parallel num_threads(ncores) private(tmpvec)
        {
            tmpvec = (double *)calloc(ngene, sizeof(double)); // zero initialized
            #pragma omp for collapse(2)
            for (x=csize; x<dimsp[0]-csize; x++) {
                for (y=csize; y<dimsp[1]-csize; y++) {
                    for (i=0; i<ngene; i++)
                        tmpvec[i] = 0;
                    for (dx=-csize; dx<csize+1; dx++) {
                        for (dy=-csize; dy<csize+1; dy++) {
                            if (dx == 0 && dy == 0) continue;
                            for (i=0; i<ngene; i++)
                                tmpvec[i] += (vecs + I2D(x+dx, y+dy, dimsp[1])*ngene)[i];
                        }
                    }
                    // tmpvec[i] /= (csize * 2 + 1) * (csize * 2 + 1) - 1;
                    corrmap[I2D(x, y, dimsp[1])] = __corr__(vecs + I2D(x, y, dimsp[1])*ngene, tmpvec, ngene);
                }
            }
            free((void*)tmpvec);
        }
    } else {
        // 3D (nd == 4)
        #pragma omp parallel num_threads(ncores) private(tmpvec)
        {
            tmpvec = (double *)calloc(ngene, sizeof(double));
            #pragma omp for collapse(3)
            for (x=csize; x<dimsp[0]-csize; x++) {
                for (y=csize; y<dimsp[1]-csize; y++) {
                    for (z=csize; z<dimsp[2]-csize; z++) {
                        for (i=0; i<ngene; i++)
                            tmpvec[i] = 0;
                        for (dx=-csize; dx<csize+1; dx++) {
                            for (dy=-csize; dy<csize+1; dy++) {
                                for (dz=-csize; dz<csize+1; dz++) {
                                    if (dx == 0 && dy == 0 && dz == 0) continue;
                                    for (i=0; i<ngene; i++)
                                        tmpvec[i] += (vecs + I3D(x+dx, y+dy, z+dz, dimsp[1], dimsp[2])*ngene)[i];
                                }
                            }
                        }
                        //for (i=0; i<ngene; i++)
                        //    tmpvec[i] /= (csize * 2 + 1) * (csize * 2 + 1) * (csize * 2 + 1) - 1;
                        corrmap[I3D(x, y, z, dimsp[1], dimsp[2])] =
                            __corr__(vecs + I3D(x, y, z, dimsp[1], dimsp[2])*ngene, tmpvec, ngene);
                    }
                }
            }
            free((void*)tmpvec);
        }
    }
    Py_DECREF(arr1);

    return (PyObject *) oarr;
 fail:
    Py_XDECREF(arr1);
    return NULL;
}

static PyObject *calc_corrmap_2(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg1 = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *oarr = NULL;
    long i, k, x, y, z, dx, dy, dz;
    long nvec, nd, ngene = 0;
    double *vecs, *corrmap;
    npy_intp *dimsp;
    npy_intp dimsp2[4];
    int ncores = omp_get_max_threads();
    int csize = 1;

    static const char *kwlist[] = { "vf", "ncores", "size", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ii", const_cast<char **>(kwlist), &arg1, &ncores, &csize)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;   
    nd = PyArray_NDIM(arr1);
    if (nd != 3 && nd != 4) goto fail; // only 2D or 3D array is expected
    dimsp = PyArray_DIMS(arr1);
    for (i=0; i<nd-1; i++)
        dimsp2[i] = dimsp[i];
    dimsp2[nd-1] = (csize * 2 + 1) * (csize * 2 + 1) - 1;
    oarr = (PyArrayObject*)PyArray_ZEROS(nd, dimsp2, NPY_DOUBLE, NPY_CORDER);
    ngene = dimsp[nd-1];
    corrmap = (double *)PyArray_DATA(oarr);
    vecs = (double *)PyArray_DATA(arr1);
    nvec = 1;
    for (i=0; i<nd; i++)
        nvec *= dimsp2[i];

    // initialize corrmap with NANs
    #pragma omp parallel for num_threads(ncores)
    for (i=0; i<nvec; i++)
        corrmap[i] = NPY_NAN;

    if (nd == 3) {
        // 2D
        #pragma omp parallel for collapse(2) private(k, dx, dy)
        for (x=csize; x<dimsp[0]-csize; x++) {
            for (y=csize; y<dimsp[1]-csize; y++) {
                k = 0;
                for (dx=-csize; dx<csize+1; dx++) {
                    for (dy=-csize; dy<csize+1; dy++) {
                        if (dx == 0 && dy == 0) continue;
                        corrmap[I2D(x, y, dimsp2[1])*dimsp2[2] + (k++)] = 
                            __corr__(vecs + I2D(x, y, dimsp[1])*ngene,
                                     vecs + I2D(x+dx, y+dy, dimsp[1])*ngene, ngene);
                    }
                }
            }
        }
    } else {
        // 3D
        #pragma omp parallel for collapse(3) private(k, dx, dy, dz)
        for (x=csize; x<dimsp[0]-csize; x++) {
            for (y=csize; y<dimsp[1]-csize; y++) {
                for (z=csize; z<dimsp[2]-csize; z++) {
                    k = 0;
                    for (dx=-csize; dx<csize+1; dx++) {
                        for (dy=-csize; dy<csize+1; dy++) {
                            for (dz=-csize; dz<csize+1; dz++) {
                                if (dx == 0 && dy == 0 && dz == 0) continue;
                                corrmap[I3D(x, y, z, dimsp2[1], dimsp2[2])*dimsp2[3] + (k++)] =
                                    __corr__(vecs + I3D(x, y, z, dimsp[1], dimsp[2])*ngene, vecs + I3D(x+dx, y+dy, z+dz, dimsp[1], dimsp[2])*ngene, ngene);
                            }
                        }
                    }
                }
            }
        }
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

    static const char *kwlist[] = { "vec", "vf", "ncores", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|i", const_cast<char **>(kwlist), &arg1, &arg2, &ncores)) return NULL;
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
    {"calc_corrmap_2", (PyCFunction)calc_corrmap_2, METH_VARARGS | METH_KEYWORDS, "Creates a correlation map."},
    {"calc_kde", (PyCFunction)calc_kde, METH_VARARGS | METH_KEYWORDS, "Run kernel density estimation."},
    {"flood_fill", (PyCFunction)flood_fill, METH_VARARGS | METH_KEYWORDS, "Performs 3d flood fill based on correlation."},
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
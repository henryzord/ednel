package dn;

public class Variable {
    private String[] values;
    private float[] probabilities;

    public Variable(String[] values, float[] probabilities) {
        this.values = values;
        this.probabilities = probabilities;
    }

    public String sample() {
//        static char *kwds[] = {"a", "size", "replace", "p", NULL};
//    PyObject *size = NULL, *a = NULL, *p = NULL;
//    int replace = 1;
//
//    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OpO", kwds, &a, &size, &replace, &p)) {
//        return NULL;
//    }
//
//    if(!PyArray_Check(a)) {
//        PyErr_SetString(PyExc_TypeError, "a must be a numpy.ndarray");
//        return NULL;
//    }
//    if((p != NULL) && !PyArray_Check(p)) {
//        PyErr_SetString(PyExc_TypeError, "p must be a numpy.ndarray");
//        return NULL;
//    }
//
//    int sampled_ndims = -1;
//    npy_intp *sampled_dims;
//    if(size != NULL) {
//        if(PyTuple_Check(size)) {
//            sampled_ndims = (int)PyTuple_Size(size);
//            sampled_dims = new npy_intp [sampled_ndims];
//
//            for(int i = 0; i < sampled_ndims; i++) {
//                sampled_dims[i] = (int)PyLong_AsLong(PyTuple_GetItem(size, i));
//            }
//        } else if (PyLong_Check(size)) {
//            sampled_ndims = 1;
//            sampled_dims = new npy_intp [1];
//            sampled_dims[0] = (int)PyLong_AsLong(size);
//        } else {
//            PyErr_SetString(PyExc_TypeError, "size must be either a tuple or an integer");
//            return NULL;
//        }
//    } else {
//        PyErr_SetString(PyExc_NotImplementedError, "not implemented yet!");
//        return NULL;
//    }
//
//    int a_ndims = PyArray_NDIM((const PyArrayObject*)a),
//        p_ndims = PyArray_NDIM((const PyArrayObject*)p);
//
//    if(a_ndims > 1) {
//        PyErr_SetString(PyExc_ValueError, "a must be 1-dimensional");
//        return NULL;
//    }
//    if(p_ndims > 1) {
//        PyErr_SetString(PyExc_ValueError, "p must be 1-dimensional");
//        return NULL;
//    }
//
//    npy_intp *a_dims = PyArray_SHAPE((PyArrayObject*)a),
//             *p_dims = PyArray_SHAPE((PyArrayObject*)p);
//
//    for(int j = 0; j < a_ndims; j++) {
//        if(a_dims[j] != p_dims[j]) {
//            PyErr_SetString(PyExc_ValueError, "a and p must have the same size");
//            return NULL;
//        }
//    }
//
//    PyArray_Descr *a_descr = PyArray_DESCR((PyArrayObject*)a);
//    PyObject *sampled_obj = PyArray_NewFromDescr(
//        &PyArray_Type, a_descr,
//        sampled_ndims, sampled_dims, NULL, NULL, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, NULL
//    );
//
//    PyArrayObject *sampled = (PyArrayObject*)sampled_obj;
//
//    if(sampled == NULL) {
//        PyErr_SetString(PyExc_ValueError, "Exception ocurred while trying to allocate space for numpy array");
//        return NULL;
//    }
//
//    // type checking ends here; real code begins here
//
//    npy_intp sampled_size = PyArray_SIZE(sampled),
//             a_itemsize = PyArray_ITEMSIZE((PyArrayObject*)a),
//             p_itemsize = PyArray_ITEMSIZE((PyArrayObject*)p);
//
//    int *sampled_counters = (int*)malloc(sizeof(int) * sampled_ndims);
//    for(int j = 0; j < sampled_ndims; j++) {
//        sampled_counters[j] = 0;
//    }
//
//    char *sampled_ptr = PyArray_BYTES(sampled), *p_ptr, *a_ptr;  // data pointers
//    npy_intp sampled_itemsize = PyArray_ITEMSIZE(sampled);
//
//    int num, div;
//    float sum, spread = 1000, p_data;
//    PyObject *a_data;
//
//    for(int i = 0; i < sampled_size; i++) {
//        num = rand() % (int)spread;  // random sampled number
//        sum = 0;  // sum of probabilities so far
//
//        p_ptr = PyArray_BYTES((PyArrayObject*)p);
//        a_ptr = PyArray_BYTES((PyArrayObject*)a);
//
//        for(int k = 0; k < a_dims[0]; k++) {
//            p_data = (float)PyFloat_AsDouble(PyArray_GETITEM((PyArrayObject*)p, p_ptr));
//            a_data = PyArray_GETITEM((PyArrayObject*)a, a_ptr);
//            p_ptr += p_itemsize;
//            a_ptr += a_itemsize;
//
//            div = (int)(num/((sum + p_data) * spread));
//
//            if(div <= 0) {
//                PyArray_SETITEM(sampled, sampled_ptr, a_data);
//                break;
//            }
//            sum += p_data;
//        }
//        sampled_ptr += sampled_itemsize;
//    }
//
//    return Py_BuildValue("O", sampled);
        return null;
    }
}

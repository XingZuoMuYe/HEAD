import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport dtrsv

cpdef calculate_spline_coefficients(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y):
    cdef int nx = len(x)
    cdef np.ndarray[np.float64_t, ndim=1] b = np.zeros(nx - 1)
    cdef np.ndarray[np.float64_t, ndim=1] c = np.zeros(nx)
    cdef np.ndarray[np.float64_t, ndim=1] d = np.zeros(nx - 1)
    cdef np.ndarray[np.float64_t, ndim=1] h = np.diff(x)
    cdef np.ndarray[np.float64_t, ndim=1] a = y

    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros((nx, nx))
    cdef np.ndarray[np.float64_t, ndim=1] B = np.zeros(nx)
    cdef np.float64_t tb

    A[0, 0] = 1.0
    for i in range(nx - 1):
        if i != nx - 2:
            A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
        A[i + 1, i] = h[i]
        A[i, i + 1] = h[i]
    A[0, 1] = 0.0
    A[nx - 1, nx - 2] = 0.0
    A[nx - 1, nx - 1] = 1.0

    for i in range(nx - 2):
        B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] - 3.0 * (a[i + 1] - a[i]) / h[i]

    dtrsv("U", "N", "N", nx, A, &B[0], nx, 1)

    for i in range(nx - 1):
        d[i] = (B[i + 1] - B[i]) / (3.0 * h[i])
        tb = (a[i + 1] - a[i]) / h[i] - h[i] * (B[i + 1] + 2.0 * B[i]) / 3.0
        b[i] = tb

    return a, b, c, d

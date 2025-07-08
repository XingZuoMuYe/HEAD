"""
cubic spline planner

Author: Atsushi Sakai

"""
import math
import numpy as np
import scipy
import bisect
import time

import numpy as np
import scipy.linalg


class Spline_:
    def __init__(self, x, y):
        self.nx = len(x)
        self.b = np.zeros(self.nx - 1)
        self.c = np.zeros(self.nx)
        self.d = np.zeros(self.nx - 1)
        self.x = x
        self.y = y

        h = np.diff(x)

        # calc coefficient a
        self.a = y

        # calc coefficient c
        # calc matrix A for spline coefficient c
        A = np.zeros([self.nx, self.nx])
        A[0, 0] = 1.0
        A[self.nx - 1, self.nx - 1] = 1.0
        for i in range(self.nx - 2):
            A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        # calc matrix B for spline coefficient c
        B = np.zeros(self.nx)
        B[1:-1] = 3.0 * (self.a[2:] - self.a[1:-1]) / h[1:] - 3.0 * (self.a[1:-1] - self.a[:-2]) / h[:-1]

        # Use Numba-accelerated solver
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        self.d = (self.c[1:] - self.c[:-1]) / (3.0 * h)
        tb = (self.a[1:] - self.a[:-1]) / h - h * (self.c[1:] + 2.0 * self.c[:-1]) / 3.0
        self.b = tb

    def calc(self, t):
        # Calc position
        # if t is outside the input x, return None
        if t < self.x[0]:
            return self.a[0]
        elif t > self.x[-1]:
            return self.a[-1]
        else:
            i = np.searchsorted(self.x, t) - 1
            dx = t - self.x[i]
            result = (self.a[i]
                      + self.b[i] * dx
                      + self.c[i] * dx ** 2
                      + self.d[i] * dx ** 3)
            return result

    def calcd(self, t):
        # Calc first derivative
        # if t is outside the input x, return None
        if t < self.x[0]:
            return self.b[0]
        elif t > self.x[-1]:

            return self.b[-1]
        else:
            i = np.searchsorted(self.x, t) - 1
            dx = t - self.x[i]
            result = (self.b[i]
                      + 2.0 * self.c[i] * dx
                      + 3.0 * self.d[i] * dx ** 2)
            return result

    def calcdd(self, t):
        # Calc second derivative
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None
        else:
            i = np.searchsorted(self.x, t) - 1
            dx = t - self.x[i]
            result = (2.0 * self.c[i]
                      + 6.0 * self.d[i] * dx)
            return result

    def calcddd(self, t):
        # Calc third derivative
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None
        else:
            i = np.searchsorted(self.x, t) - 1
            result = (6.0 * self.d[i])
            return result

    def calc_A(self, h):
        # calc matrix A for spline coefficient c
        A = np.zeros([self.nx, self.nx])
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != self.nx - 2:
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]
        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def calc_B(self, h):
        # calc matrix B for spline coefficient c
        B = np.zeros([self.nx, 1])
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1] - \
                       3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B


class Spline2D_:
    def __init__(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        self.s = np.append([0], np.cumsum(self.ds))
        self.sx = Spline_(self.s, x)
        self.sy = Spline_(self.s, y)

    def calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = np.append([0], np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        # calc position
        x = self.sx.calc(s)
        y = self.sy.calc(s)
        return x, y

    def calc_curvature(self, s):
        # calc curvature
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)

        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** 1.5)
        return k

    def calc_d_curvature(self, s):
        # calc curvature d
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dddx = self.sx.calcddd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        dddy = self.sy.calcddd(s)

        a = dx * ddy - dy * ddx
        b = dx * dddy - dy * dddx
        c = dx * ddx + dy * ddy
        d = dx * dx + dy * dy

        dk = (b * d - 3.0 * a * c) / (d * d * d)
        return dk

    def calc_yaw(self, s):
        # calc yaw
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)

        yaw = np.arctan2(dy, dx)
        return yaw


class Spline3D:
    """
    3D Cubic Spline class
    """

    def __init__(self, x, y, z):
        # time1 = time.time()
        self.s = self.__calc_s(x, y, z)
        # time2 = time.time()
        self.sx = Spline_(self.s, x)
        # time3 = time.time()
        self.sy = Spline_(self.s, y)
        # time4 = time.time()
        self.sz = Spline_(self.s, z)
        # time5 = time.time()

    def __calc_s(self, x, y, z):
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2 + idz ** 2) for (idx, idy, idz) in zip(dx, dy, dz)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        # print('s_min:', s[0])
        # print('s_max:', s[-1])
        return s
        # 计算路径点的s坐标

    def calc_position(self, s):
        u"""
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)
        z = self.sz.calc(s)
        return x, y, z

    def calc_curvature(self, s):
        u"""
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        delta = 0.0
        if dx <= 0.0:
            if dy <= 0.0:
                dx = -dx
                dy = -dy
                delta = -np.pi
            else:
                delta = np.pi

        yaw = np.arctan(dy / dx) + delta

        return yaw

    def calc_pitch(self, s):
        """
        calc pitch - this function needs to be double checked
        """
        dx = self.sx.calcd(s)
        dz = self.sz.calcd(s)
        pitch = math.atan2(dz, dx)
        return pitch

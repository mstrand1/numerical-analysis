import numpy as np
from nanalysis.num_methods import NumMethods


class Interpolation:
    def __init__(self, func=None):
        self.func = func

    def lagrange(self, x, approx):
        n = len(x)
        interp = 0
        l = np.zeros(n)
        for i in range(n):
            l[i] = 1
            for k in range(n):
                if k != i:
                    l[i] *= (approx - x[k]) / (x[i] - x[k])
            interp += self.func(x[i]) * l[i]
        return interp

    def dd_lagrange(self, x, approx):
        dd = self.div_diff(x)
        interp = dd[0, 0]
        n = len(x)
        for i in range(1, n):
            d = 1
            for j in range(i):
                d *= (approx - x[j])
            interp += d * dd[i, i]
        return interp

    def div_diff(self, x):
        n = len(x)
        dd = np.zeros((n, n))
        dd[:, 0] = np.transpose(self.func(x))
        for i in range(n-1):
            for j in range(i+1):
                dd[i+1, j+1] = (dd[i+1, j] - dd[i, j]) / (x[i+1] - x[i-j])
        return dd

    def hdiv_diff(self, x):
        n = len(x)
        z = np.zeros((2*n))
        q = np.zeros((2*n, 2*n))
        for i in range(n):
            z[2*i] = x[i]
            z[2*i+1] = x[i]
            q[2*i, 0] = self.func(x[i])
            q[2*i+1, 0] = self.func(x[i])
            q[2*i+1, 1] = NumMethods(self.func).mid_diff(x[i])
            if i != 0:
                q[2*i, 1] = (q[2*i, 0] - q[2*i-1, 0]) / (z[2*i]-z[2*i-1])
        for i in range(2, 2*n):
            for j in range(2, i+1):
                q[i, j] = (q[i, j-1] - q[i-1, j-1]) / (z[i]-z[i-j])
        return q, z

    def hermite(self, x, approx):
        hdd, z = self.hdiv_diff(x)
        n = int(len(z)/2)
        hp = hdd[0, 0]
        for i in range(1, 2*n):
            d = 1
            for j in range(i):
                d *= (approx - z[j])
            hp += d * hdd[i, i]
        return hp

    def spline_coeff(self, x):
        n = len(x)
        a = np.zeros(n)
        for i in range(n):
            a[i] = self.func(x[i])
        n = len(x)
        h = np.zeros(n)
        alph = h.copy()
        mu = h.copy()
        l = a.copy()
        z = l.copy()
        c = l.copy()
        b = h.copy()
        d = h.copy()
        for i in range(n-1):
            h[i] = x[i+1] - x[i]
        for i in range(1, n-1):
            alph[i] = (3/h[i]) * (a[i+1] - a[i]) - (3/h[i-1]) * (a[i] - a[i-1])
        l[0] = 1
        mu[0] = 0
        z[0] = 0
        for i in range(1, n-1):
            l[i] = 2*(x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
            mu[i] = h[i] / l[i]
            z[i] = (alph[i] - h[i-1] * z[i-1]) / l[i]
        l[n-1] = 1
        z[n-1] = 0
        c[n-1] = 0
        for j in range(n-2, -1, -1):
            c[j] = z[j] - mu[j] * c[j+1]
            b[j] = ((a[j+1] - a[j]) / h[j]) - (h[j] * (c[j+1] + 2*c[j])/3)
            d[j] = (c[j+1] - c[j]) / (3*h[j])
        a = a[:n]
        b = b[:n-1]
        c = c[:n-1]
        d = d[:n-1]
        return a, b, c, d

    def cubic_spline(self, x, approx):
        n = len(x)
        a, b, c, d = self.spline_coeff(x)
        if type(approx) == float:
            approx = [approx]
        m = len(approx)
        sy = np.zeros(m)
        for j in range(n):
            for i in range(m):
                if x[j] < approx[i] < x[j+1]:
                    sy[i] = a[j] + b[j]*(approx[i] - x[j]) + c[j]*(approx[i] - x[j])**2 + d[j]*(approx[i] - x[j])**3
        return sy

    def cheby_coeff(self, n):
        f = self.func
        bp = lambda a: f(np.cos(a))
        a = np.zeros(n+2)
        a[1] = (1/np.pi) * NumMethods(bp).simp_comp(0, np.pi, 50)
        for i in range(1, n+1):
            g = lambda x: (f(np.cos(x)) * np.cos(i*x))
            a[i+1] = (2/np.pi) * NumMethods(g).simp_comp(0, np.pi, 50)
        return a[1:n+2]

    def chebyshev(self, approx, n=4):
        c = self.cheby_coeff(n)
        p = 0
        for i in range(n+1):
            t = lambda a: np.cos(i * np.arccos(a))
            p += c[i] * t(approx)
        return p

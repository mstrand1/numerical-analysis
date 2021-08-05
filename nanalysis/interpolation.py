import numpy as np
from nanalysis.num_methods import NumMethods


class Interpolation:
    """
    Methods of approximation by different polynomial interpolants using either a known f(x)
    or a set of x,y values.
    """

    def __init__(self, func=None):
        self.func = func

    def lagrange(self, x, y=None, approx=None):
        """
          Approximation via Lagrange polynomial interpolation.

          Args:
              x (1D array): x,y interpolation point.
              y (1D array): x,y interpolation point.
              approx (float): Value to approximate.

          Returns:
              float: Approximation
          """
        n = len(x)
        if not y:
            y = np.zeros(n)
            for i in range(n):
                y[i] = self.func(x[i])
        interp = 0
        l = np.zeros(n)
        for i in range(n):
            l[i] = 1
            for k in range(n):
                if k != i:
                    l[i] *= (approx - x[k]) / (x[i] - x[k])
            interp += y[i] * l[i]
        return interp

    def dd_lagrange(self, x, y=None, approx=None):
        """
          Approximation via Lagrange polynomial interpolation with divided-difference coefficients.

          Args:
              x (1D array): x,y interpolation point.
              y (1D array): x,y interpolation point.
              approx (float): Value to approximate.

          Returns:
              float: Approximation
          """
        n = len(x)
        if not y:
            y = np.zeros(n)
            for i in range(n):
                y[i] = self.func(x[i])
        dd = self.div_diff(x, y)
        interp = dd[0, 0]
        for i in range(1, n):
            d = 1
            for j in range(i):
                d *= (approx - x[j])
            interp += d * dd[i, i]
        return interp

    def div_diff(self, x, y=None):
        """
          Obtains Newton's divided-difference coefficients for the interpolating polynomial.

          Args:
              x (1D array): x,y interpolation point
              y (1D array): x,y interpolation point

          Returns:
              ndarray: 2D (n x n) array of divided-difference coefficients. ith column represents
              i-1th divided differences.
          """
        n = len(x)
        dd = np.zeros((n, n))
        if not y:
            dd[:, 0] = np.transpose(self.func(x))
        elif y:
            dd[:, 0] = np.transpose(y)
        for i in range(n-1):
            for j in range(i+1):
                dd[i+1, j+1] = (dd[i+1, j] - dd[i, j]) / (x[i+1] - x[i-j])
        return dd

    def hdiv_diff(self, x, y=None, yp=None):
        """
          Obtains coefficients of Hermite interpolating polynomial using divided-differences.
          Requires equal-length x,y,yp arrays or just x array if f(x) is known and differentiable.

          Args:
              x (1D array): x,y interpolation point
              y (1D array): x,y interpolation point
              yp (1D array): First derivative y' evaluated at x

          Returns:
              ndarray: 2D (2n x 2n) array of divided-difference coefficients. ith column represents
              i-1th divided differences.
          """
        n = len(x)
        z = np.zeros((2*n))
        q = np.zeros((2*n, 2*n))
        if not y and not yp:
            for i in range(n):
                y[i] = self.func(x[i])
                yp[i] = NumMethods(self.func).mid_diff(x[i])
        for i in range(n):
            z[2*i] = x[i]
            z[2*i+1] = x[i]
            q[2*i, 0] = y[i]
            q[2*i+1, 0] = y[i]
            q[2*i+1, 1] = yp[i]
            if i != 0:
                q[2*i, 1] = (q[2*i, 0] - q[2*i-1, 0]) / (z[2*i]-z[2*i-1])
        for i in range(2, 2*n):
            for j in range(2, i+1):
                q[i, j] = (q[i, j-1] - q[i-1, j-1]) / (z[i]-z[i-j])
        return q, z

    def hermite(self, x, y=None, yp=None, approx=None):
        """
          Gives approximation using Hermite polynomial with divided-difference coefficients.
          Requires equal-length x,y,yp arrays or just x array if f(x) is known and differentiable.

          Args:
              x (1D array): x,y interpolation point.
              y (1D array): x,y interpolation point.
              yp (1D array): First derivative y' evaluated at x
              approx (float): Value to approximate.

          Returns:
              float: Hermite approximate.
          """
        hdd, z = self.hdiv_diff(x,y,yp)
        n = int(len(z)/2)
        hp = hdd[0, 0]
        for i in range(1, 2*n):
            d = 1
            for j in range(i):
                d *= (approx - z[j])
            hp += d * hdd[i, i]
        return hp

    def spline_coeff(self, x, y=None):
        """
          Construct cubic spline coefficients to approximate on appropriate interval(s).
          Ex: for x in [x(i-1), x(i)], spline S_i(x) = a_i + b_ix + c_ix^2 + d_ix^3

          Args:
              x (1D array): x,y interpolation point.
              y (1D array): x,y interpolation point.

          Returns:
              1D array: a, b, c, d are the spline coefficients: a + bx +cx^2 +dx^3
          """
        n = len(x)
        a = np.zeros(n)
        if not y:
            for i in range(n):
                a[i] = self.func(x[i])
        elif y:
            a = y
        h, alph, mu, l, z, c, b, d = np.zeros((8, n))
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
        a = a[:n-1]
        b = b[:n-1]
        c = c[:n-1]
        d = d[:n-1]
        return a, b, c, d

    def cubic_spline(self, x, y=None, approx=None):
        """
          Construct cubic splines to approximate on appropriate interval(s).

          Args:
              x (1D array): x,y interpolation point.
              y (1D array): x,y interpolation point.
              approx (float or 1D array): Value(s) to approximate.

          Returns:
              ndarray: 1D array of approximations.
          """
        n = len(x)
        a, b, c, d = self.spline_coeff(x, y)
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
        """
          Compute Chebyshev coefficients for Chebyshev polynomial approximations.
          Requires known function f(x).

          Args:
              n (int): Degree of Chebyshev polynomial.

          Returns:
              ndarray: 1D array of coefficients to be used in polynomial approximation.
          """
        f = self.func
        bp = lambda a: f(np.cos(a))
        a = np.zeros(n+2)
        a[1] = (1/np.pi) * NumMethods(bp).simp_comp(0, np.pi, 50)
        for i in range(1, n+1):
            g = lambda x: (f(np.cos(x)) * np.cos(i*x))
            a[i+1] = (2/np.pi) * NumMethods(g).simp_comp(0, np.pi, 50)
        return a[1:n+2]

    def chebyshev(self, approx, n=4):
        """
          Compute Chebyshev polynomial approximation of f(x).

          Args:
              n (int): Max degree Chebyshev polynomial.

          Returns:
              float: Approximation to f(x) at approx value.
          """
        c = self.cheby_coeff(n)
        p = 0
        for i in range(n+1):
            t = lambda a: np.cos(i * np.arccos(a))
            p += c[i] * t(approx)
        return p

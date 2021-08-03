import numpy as np


class NumericalAnalysis:
    def __init__(self, func=None):
        self.func = func

    def for_diff(self, x, h=10**(-5)):
        return (self.func(x + h) - self.func(x))/h

    def mid_diff(self, x, h=10**(-5)):
        return (self.func(x + h) - self.func(x - h))/(2*h)

    def second_diff(self, x, h=10**(-5)):
        return (self.func(x - h) - 2*self.func(x) + self.func(x+h)) / (h**2)

    def newton_method(self, p_0=0.0, n_0=10, tol=10**(-7)):
        i = 1
        found = False
        while i < n_0:
            p = p_0 - self.func(p_0) / self.mid_diff(p_0)
            if abs(p - p_0) < tol:
                found = True
                break
            i += 1
            p_0 = p
        if not found:
            print("Root not found after", n_0," iterations.")
            return
        elif found:
            print("Root: ", p, " found after ", i+1, "iterations.")
            return p

    def bisection_method(self, a=-5.0, b=5.0, n_0=10, tol=10**(-7)):
        if self.func(a) * self.func(b) > 0:
            print("Bad interval: f must change signs.")
            return None
        for i in range(n_0):
            p = (a + b) / 2
            if abs(b - a) < tol:
                print("Invalid interval: too small")
                break
            fp = self.func(p)
            if abs(fp) < tol:
                print("Root found at", p, "after ", i+1, "iterations.")
                return p
            if self.func(a) * fp < 0:
                b = p
            else:
                a = p
        print("Failed to find root after", n_0, "iterations.")
        return None

    def fixed_point(self, p_0=0.0, n_0=10, tol=10**(-7)):
        i = 1
        found = False
        while i < n_0:
            p = self.func(p_0)
            if abs(p - p_0) < tol:
                found = True
                break
            i += 1
            p_0 = p
        if not found:
            print("Fixed-point not found after", n_0," iterations.")
        elif found:
            print("Root: ", p, " found after ", i+1, "iterations.")

    def gauss_elim(self, a):
        n = len(a)
        x = np.zeros(n)
        back_sum = 0
        for i in range(n):
            while a[min(range(i, n)), i] == 0:
                k = i + 1
                if k == n + 1:
                    print("No unique solution.")
                    return None
                if min(range(k, n)) != i:
                    temp = a[i, :]
                    a[i, :] = a[min(range(k, n)), :]
                    a[min(range(k, n)), :] = temp
            for j in range(i+1, n):
                m_ji = a[j, i] / a[i, i]
                a[j, :] = a[j, :] - m_ji * a[i, :]
            print(a)
            print()
        if a[n-1, n-1] == 0:
            print("No unique solution.")
            return None
        x[n-1] = a[n-1, n]/a[n-1, n-1]
        for i in range(n-2, -1, -1):
            for j in range(i+1, n):
                back_sum = back_sum + x[j] * a[i, j]
            x[i] = ((a[i, n] - back_sum)/a[i, i])
            back_sum = 0
        print("(x1, x2, ... , xn) = ")
        return x

    def rref(self, a):
        a = np.array(a)
        n = len(a)
        for i in range(n):
            while a[min(range(i, n)), i] == 0:
                k = i + 1
                if k == n:
                    print("Matrix not Invertible")
                    return
                if min(range(k, n)) != i:
                    temp = a[i, :]
                    a[i, :] = a[min(range(k, n)), :]
                    a[min(range(k, n)), :] = temp
            for j in range(i):
                m_ji = a[j, i] / a[i, i]
                a[j, :] = a[j, :] - m_ji * a[i, :]
            for j in range(i+1, n):
                m_ji = a[j, i] / a[i, i]
                a[j, :] = a[j, :] - m_ji * a[i, :]
        if a[n-1, n-1] == 0:
            print("Matrix not Invertible")
            return
        for j in range(n):
            a[j, :] = a[j, :] / a[j, j]
        return a.round(5)

    def aug(self, a):
        return np.append(a, np.identity(len(a)), axis=1)

    def inv_matrix(self, a):
        n = len(a)
        inv = self.rref(self.aug(a))
        return inv[:, n:2*n]

    @staticmethod
    def jacobi(self, a, b, x_0=None, tol=10**(-7), n_0=15):
        n = len(a)
        if not x_0:
            x_0 = np.zeros(n)
        x = x_0.copy()
        k = 1
        s = x.copy()
        while k < n_0:
            for i in range(n):
                for j in range(n):
                    if j != i:
                        s[i] += a[i, j] * x_0[j]
                x[i] = (b[i] - s[i]) / a[i, i]
                s[i] = 0
            if np.linalg.norm((x - x_0), ord=np.inf) < tol:
                print("Solution found after ", k, "iterations.")
                return x
            k += 1
            x_0 = x.copy()
        print("Max iterations", n_0, " exceeded")
        return None

    @staticmethod
    def gauss_seidel(self, a, b, x_0=None, tol=10**(-7), n_0=15):
        n = len(a)
        if not x_0:
            x_0 = np.zeros(n)
        x = x_0.copy()
        sum_l = x_0.copy()
        sum_u = x_0.copy()
        k = 1
        while k < n_0:
            for i in range(n):
                for j in range(i+1, n):
                    sum_u[i] += a[i, j] * x_0[j]
                for j in range(i):
                    sum_l[i] += a[i, j] * x[j]
                x[i] = (b[i] - sum_l[i] - sum_u[i]) / a[i, i]
                sum_u[i] = 0
                sum_l[i] = 0
            if np.linalg.norm(x-x_0, ord=np.inf) < tol:
                print("Solution found after ", k, "iterations.")
                return x
            k += 1
            x_0 = x.copy()
        print("Max iterations", n_0, " exceeded.")
        return None

    # somethings wrong
    def SOR(self, a, b, x_0=None, tol=10**(-5), n_0=25, w=1.00):
        n = len(a)
        if not x_0:
            x_0 = np.zeros(n)
        x = x_0.copy()
        k = 1
        sum_l = x_0.copy()
        sum_u = x_0.copy()
        while k < n_0:
            for i in range(n):
                for j in range(i + 1, n):
                    sum_u[i] += a[i, j] * x_0[j]
                for j in range(i):
                    sum_l[i] += a[i, j] * x[j]
                x[i] = ((1.0-w) * x_0[i]) + ((w * b[i] - (sum_l[i] + sum_u[i])) / a[i, i])
                sum_u[i] = 0.0
                sum_l[i] = 0.0
            if np.linalg.norm(x - x_0, ord=np.inf) < tol:
                print("Solution found after ", k, "iterations.")
                return x
            k += 1
            x_0 = x.copy()
        print("Max iterations", n_0, " exceeded.")
        return None

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
        dd = self.divdiff(x)
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
            q[2*i+1, 1] = self.mid_diff(x[i])
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

    def trapezoid_rule(self, a, b):
        return ((b - a)/2) * (self.func(a) + self.func(b))

    def trap_comp(self, a, b, n=10):
        h = (b - a)/n
        m = 0
        for i in range(n):
            m += self.func(a+i*h)
        return (h/2) * (self.func(a) + 2*m + self.func(b))

    def simpsons_rule(self, a, b):
        h = (b - a)/2
        return (h/3) * (self.func(a) + 4*self.func((b+a)/2) + self.func(b))

    def simp_comp(self, a, b, n=10):
        h = (b-a)/n
        nn0 = self.func(a) + self.func(b)
        nn1 = 0
        nn2 = 0
        for i in range(1, n):
            x = a+i*h
            if i % 2 == 0:
                nn2 += self.func(x)
            else:
                nn1 += self.func(x)
        xi = h*(nn0 + 2*nn2 + 4*nn1)/3
        return xi

    #not working
    def simp_adpt(self, ap, bp, tol=10**(-5), n_0=20):
        approx = 0
        i = 0
        e = np.zeros(n_0)
        a = e.copy()
        h = e.copy()
        fa = e.copy()
        fc = e.copy()
        fb = e.copy()
        s = e.copy()
        l = e.copy()
        fd = 0
        fe = 0
        s1 = 0
        s2 = 0
        v1 = 0
        v2 = 0
        v3 = 0
        v4 = 0
        v5 = 0
        v6 = 0
        v7 = 0
        v8 = 0
        e[i] = 10*tol
        a[i] = ap
        h[i] = (bp-ap)/2
        fa[i] = self.func(ap)
        fc[i] = self.func(ap+h[i])
        fb[i] = self.func(bp)
        s[i] = h[i] * (fa[i] + 4*fc[i] + fb[i])/3
        l[i] = 1
        while i > 0:
            fd = self.func(a[i] + h[i]/2)
            fe = self.func(a[i] + 3*h[i]/2)
            s1 = h[i]*(fa[i] + 4*fd + fc[i])/6
            s2 = h[i]*(fc[i]+4*fe + fb[i])/6
            v1 = a[i]
            v2 = fa[i]
            v3 = fc[i]
            v4 = fb[i]
            v5 = h[i]
            v6 = e[i]
            v7 = s[i]
            v8 = l[i]
            i -= 1

        if abs(s1 + s2 - v7) < v6:
            print('mommy')
            approx += s1 + s2
        elif v8 >= n_0:
            print("Level exceeded.")
            return None
        else:
            i += 1
            a[i] = v1 + v5
            fa[i] = v3
            fc[i] = fe
            fb[i] = v4
            h[i] = v5/2
            e[i] = v6/2
            s[i] = s2
            l[i] = v8 + 1

            i += 1

            a[i] = v1
            fa[i] = v2
            fc[i] = fd
            fb[i] = v3
            h[i] = h[i-1]
            e[i] = e[i-1]
            s[i] = s1
            l[i] = l[i-1]

        return approx

    def gquad(self, a, b):
        return (self.func((1 / 2)*((b-a)*(-np.sqrt(3)/3)+a+b))+self.func((1/2)*((b-a)*(np.sqrt(3)/3)+a+b)))*(b-a)/2

    def adpt_gquad(self, a, b, level=0, sm=0, n_0=20, tol=10**(-7)):
        level += 1
        one_gauss = self.gquad(a, b)
        c = (a+b)/2
        two_gauss = self.gquad(a, c) + self.gquad(c, b)
        if level > n_0:
            print("Error: Max depth reached.")
        else:
            if abs(one_gauss - two_gauss) < tol:
                sm += two_gauss
            else:
                sm = self.aquad(a, c, level=level, sm=sm, n_0=n_0)
                sm = self.aquad(c, b, level=level, sm=sm, n_0=n_0)
        return sm

    def cheby_coeff(self, n):
        f = self.func
        bp = lambda a: f(np.cos(a))
        a = np.zeros(n+2)
        a[1] = (1/np.pi) * NumericalAnalysis(bp).simp_comp(0, np.pi, 50)
        for i in range(1, n+1):
            g = lambda x: (f(np.cos(x)) * np.cos(i*x))
            a[i+1] = (2/np.pi) * NumericalAnalysis(g).simp_comp(0, np.pi, 50)
        return a[1:n+2]

    def chebyshev(self, approx, n=4):
        c = self.cheby_coeff(n)
        p = 0
        for i in range(n+1):
            t = lambda a: np.cos(i * np.arccos(a))
            p += c[i] * t(approx)
        return p

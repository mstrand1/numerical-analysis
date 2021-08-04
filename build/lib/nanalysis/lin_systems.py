import numpy as np


class LinSys:
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
            for j in range(i +1, n):
                m_ji = a[j, i] / a[i, i]
                a[j, :] = a[j, :] - m_ji * a[i, :]
            print(a)
            print()
        if a[ n -1, n- 1] == 0:
            print("No unique solution.")
            return None
        x[n - 1] = a[n - 1, n] / a[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                back_sum = back_sum + x[j] * a[i, j]
            x[i] = ((a[i, n] - back_sum) / a[i, i])
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
            for j in range(i + 1, n):
                m_ji = a[j, i] / a[i, i]
                a[j, :] = a[j, :] - m_ji * a[i, :]
        if a[n - 1, n - 1] == 0:
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
        return inv[:, n:2 * n]

    @staticmethod
    def jacobi(self, a, b, x_0=None, tol=10 ** (-7), n_0=15):
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
    def gauss_seidel(self, a, b, x_0=None, tol=10 ** (-7), n_0=15):
        n = len(a)
        if not x_0:
            x_0 = np.zeros(n)
        x = x_0.copy()
        sum_l = x_0.copy()
        sum_u = x_0.copy()
        k = 1
        while k < n_0:
            for i in range(n):
                for j in range(i + 1, n):
                    sum_u[i] += a[i, j] * x_0[j]
                for j in range(i):
                    sum_l[i] += a[i, j] * x[j]
                x[i] = (b[i] - sum_l[i] - sum_u[i]) / a[i, i]
                sum_u[i] = 0
                sum_l[i] = 0
            if np.linalg.norm(x - x_0, ord=np.inf) < tol:
                print("Solution found after ", k, "iterations.")
                return x
            k += 1
            x_0 = x.copy()
        print("Max iterations", n_0, " exceeded.")
        return None

    # somethings wrong
    def SOR(self, a, b, x_0=None, tol=10 ** (-5), n_0=25, w=1.00):
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
                x[i] = ((1.0 - w) * x_0[i]) + ((w * b[i] - (sum_l[i] + sum_u[i])) / a[i, i])
                sum_u[i] = 0.0
                sum_l[i] = 0.0
            if np.linalg.norm(x - x_0, ord=np.inf) < tol:
                print("Solution found after ", k, "iterations.")
                return x
            k += 1
            x_0 = x.copy()
        print("Max iterations", n_0, " exceeded.")
        return None

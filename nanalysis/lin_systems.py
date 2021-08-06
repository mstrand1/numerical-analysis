import numpy as np


class LinSys:
    """
    Methods for solving linear systems of the form Ax=b.
    """

    def gauss_elim(self, a):
        """
        Performs Gaussian elimination followed by back substitution to solve a system Ax = b.

        Args:
            a (ndarray): Augmented (n x n+1) 2D array [A : b].

        Returns:
            ndarray: 1D solutions array (x)
        """
        n = len(a)
        x = np.zeros(n)
        back_sum = 0
        for i in range(n):
            while a[min(range(i, n)), i] == 0:
                k = i + 1
                if k == n + 1:
                    print("No unique solution")
                    return None
                if min(range(k, n)) != i:
                    temp = a[i, :]
                    a[i, :] = a[min(range(k, n)), :]
                    a[min(range(k, n)), :] = temp
            for j in range(i+1, n):
                m_ji = a[j, i] / a[i, i]
                a[j, :] = a[j, :] - m_ji * a[i, :]
        if a[n-1, n-1] == 0:
            print("No unique solution")
            return None
        x[n-1] = a[n-1, n] / a[n-1, n-1]
        for i in range(n-2, -1, -1):
            for j in range(i+1, n):
                back_sum = back_sum + x[j] * a[i, j]
            x[i] = ((a[i, n] - back_sum) / a[i, i])
            back_sum = 0
        print("(x1, x2, ... , xn) = ")
        return x

    def __rref(self, a):
        """
        Transforms a matrix into its reduced row echelon form.

        Args:
            a (ndarray): Augmented (n x n) 2D array. Last column is b.

        Returns:
            ndarray: 1D solutions array (x).
        """
        a = np.array(a)
        n = len(a)
        for i in range(n):
            while a[min(range(i, n)), i] == 0:
                k = i + 1
                if k == n:
                    print("Matrix not invertible")
                    return None
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
            print("Matrix cannot be put in rref")
            return None
        for j in range(n):
            a[j, :] = a[j, :] / a[j, j]
        return a.round(5)

    def __aug(self, a):
        """
        Augments a matrix with the identity matrix.

        Args:
            a (ndarray): Augmented (n x n) 2D array.

        Returns:
            ndarray: 2D (n x n+n) array.
        """
        return np.append(a, np.identity(len(a)), axis=1)

    def inv_matrix(self, a):
        """
        Calculates the inverse of a matrix A through row reduction of [A : I].

        Args:
            a (ndarray): Augmented (n x n) 2D array.

        Returns:
            ndarray: Columns n+1:2n of [I : A^-1].
        """
        n = len(a)
        inv = self.__rref(self.__aug(a))
        return inv[:, n:2 * n]

    def jacobi(self, a, b, x_0=None, tol=10**(-7), n_0=15):
        """
        Jacobi iterative method for solving a nxn linear system Ax = b.

        Args:
            a (ndarray): A - (n x n) 2D array.
            b (ndarray): b - (n x 1) 2D array.
            x_0 (ndarray): Initial approximation vector
            tol (float): Error tolerance
            n_0 (int): Max iterations

        Returns:
            ndarray: 1D solutions solution array (x).
        """
        n = len(a)
        if x_0 is None:
            x_0 = np.zeros(n)
        x = x_0.copy()
        k = 1
        s = x.copy()
        while k <= n_0:
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
        return x

    def gauss_seidel(self, a, b, x_0=None, tol=10**(-7), n_0=15):
        """
        Gauss-Seidel iterative technique for solving a nxn linear system Ax = b.

        Args:
            a (ndarray): A - (n x n) 2D array
            b (ndarray): b - (n x 1) 2D array
            x_0 (ndarray): Initial approximation vector
            tol (float): Error tolerance
            n_0 (int): Max iterations

        Returns:
            ndarray: 1D solutions solution array (x)
        """
        n = len(a)
        if x_0 is None:
            x_0 = np.zeros(n)
        x = x_0.copy()
        sum_lower = x_0.copy()
        sum_upper = x_0.copy()
        k = 1
        while k <= n_0:
            for i in range(n):
                for j in range(i + 1, n):
                    sum_upper[i] += a[i, j] * x_0[j]
                for j in range(i):
                    sum_lower[i] += a[i, j] * x[j]
                x[i] = (b[i] - sum_lower[i] - sum_upper[i]) / a[i, i]
                sum_upper[i] = 0
                sum_lower[i] = 0
            if np.linalg.norm(x - x_0, ord=np.inf) < tol:
                print("Solution found after ", k, "iterations.")
                return x
            k += 1
            x_0 = x.copy()
        print("Max iterations", n_0, " exceeded")
        return x

    def sor(self, a, b, x_0=None, w=1.10, tol=10**(-5), n_0=25):
        """
        Successive over-relaxation (SOR) method for solving a nxn linear system Ax = b.

        Args:
            a (ndarray): A, (n x n) 2D array.
            b (ndarray): b, (1 x n) 1D array.
            x_0 (ndarray): (1 x n) 1D array - Initial approximation vector
            tol (float): Error tolerance
            n_0 (int): Max iterations
            w (float): Weight. 1 < w for accelerated convergence.

        Returns:
            ndarray: 1D solutions solution array (x).
        """
        n = len(a)
        if x_0 is None:
            x_0 = np.zeros(n)
        x = x_0.copy()
        k = 1
        sum_lower = x_0.copy()
        sum_upper = x_0.copy()
        while k <= n_0:
            for i in range(n):
                for j in range(i + 1, n):
                    sum_upper[i] += a[i, j] * x_0[j]
                for j in range(i):
                    sum_lower[i] += a[i, j] * x[j]
                x[i] = ((1.0 - w) * x_0[i]) + ((w * (b[i] - sum_lower[i] - sum_upper[i])) / a[i, i])
                sum_upper[i] = 0.0
                sum_lower[i] = 0.0
            if np.linalg.norm(x - x_0, ord=np.inf) < tol:
                print("Solution found after ", k, "iterations.")
                return x
            k += 1
            x_0 = x.copy()
        print("Max iterations", n_0, " exceeded")
        return x

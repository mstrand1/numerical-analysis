from nanalysis.num_methods import NumMethods


class FindRoot:
    """
    Methods for finding roots and fixed-points of a function f(x).
    """

    def __init__(self, func=None):
        """
        func (function): Function f(x).
        """
        self.func = func

    def newton_method(self, p_0=0.0, n_0=10, tol=10**(-7)):
        """
          Uses Newton's method to approximate root to f(x) given an initial approximation.

          Args:
              p_0 (float): Initial approximation to root.
              n_0 (int): Max iterations.
              tol (float): Error tolerance.

          Returns:
              float: Approximation to root f(x) = 0.
          """
        i = 1
        found = False
        while i < n_0:
            p = p_0 - self.func(p_0) / NumMethods(self.func).mid_diff(p_0)
            if abs(p - p_0) < tol:
                found = True
                break
            i += 1
            p_0 = p
        if not found:
            print("Root not found after", n_0, " iterations.")
            return
        elif found:
            print("Root: ", p, " found after ", i+ 1, "iterations.")
            return p

    def bisection_method(self, a=-5.0, b=5.0, n_0=10, tol=10**(-7)):
        """
          Uses Bisection method to approximate root to f(x) on interval [a,b].

          Args:
              a,b (float): Defines interval [a,b] to search for root.
              n_0 (int): Max iterations.
              tol (float): Error tolerance.

          Returns:
              float: Approximation to root f(x) = 0.
          """
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
                print("Root found at", p, "after ", i + 1, "iterations.")
                return p
            if self.func(a) * fp < 0:
                b = p
            else:
                a = p
        print("Failed to find root after", n_0, "iterations.")
        return None

    def fixed_point(self, p_0=0.0, n_0=10, tol=10**(-7)):
        """
          Finds p, the solution to p = g(p).

          Args:
              p_0 (float): Initial approximation
              n_0 (int): Max iterations.
              tol (float): Error tolerance.

          Returns:
              float: Approximation to fixed point p = g(p).
          """
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
            print("Fixed-point not found after", n_0, " iterations.")
            return None
        elif found:
            print("Root: ", p, " found after ", i + 1, "iterations.")
            return p

from nanalysis.num_methods import NumMethods


class FindRoot:
    def __init__(self, func=None):
        self.func = func

    def newton_method(self, p_0=0.0, n_0=10, tol=10**(-7)):
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
        elif found:
            print("Root: ", p, " found after ", i + 1, "iterations.")

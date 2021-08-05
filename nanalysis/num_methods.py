import numpy as np


class NumMethods:
    """
    Methods of numerical differentiation and integration.
    """
    def __init__(self, func=None):
        self.func = func

    def for_diff(self, x, h=10**(-5)):
        """
          Forward Differences: Approximates f'(x) using (f(x+h)-f(x))/h for very small h.

          Args:
              x (float): Approximation point
              h (float): Increment

          Returns:
              float: f'(x) approximation
          """
        return (self.func(x + h) - self.func(x))/h

    def end_3diff(self, x, h=10**(-5)):
        """
          Three-Point Endpoint: Approximates f'(x) using
          (-3f(x)+4f(x+h)-f(x+2h))/2h for very small h.

          Args:
              x (float): Approximation point
              h (float): Increment

          Returns:
              float: f'(x) approximation
          """
        return (-3*self.func(x) + 4*self.func(x+h) - self.func(x+2*h))/(2*h)

    def mid_3diff(self, x, h=10**(-5)):
        """
          Three-Point Midpoint: Approximates f'(x) using (f(x+h)-f(x-h))/2h for very small h.

          Args:
              x (float): Approximation point
              h (float): Increment

          Returns:
              float: f'(x) approximation
          """
        return (self.func(x + h) - self.func(x - h))/(2*h)

    def second_diff(self, x, h=10**(-5)):
        return (self.func(x - h) - 2*self.func(x) + self.func(x+h)) / (h**2)

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
                sm = self.adpt_gquad(a, c, level=level, sm=sm, n_0=n_0)
                sm = self.adpt_gquad(c, b, level=level, sm=sm, n_0=n_0)
        return sm

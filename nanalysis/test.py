from nanalysis import NumMethods, LinSys, Interpolation, FindRoot
import numpy as np

f = lambda x: np.cos(2*x)*np.exp(-x)
b = NumMethods(f)
print(b.simp_adpt(0, 2*np.pi, tol=0.5*10**(-4), n_0=10))
# 108.555281

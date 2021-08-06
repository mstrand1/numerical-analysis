from nanalysis import NumMethods, LinSys, Interpolation, FindRoot
import numpy as np

f = lambda x: np.exp(2*x)*np.sin(3*x)
b = NumMethods(f)
print(b.simp_adpt(1, 3))
# 108.555281

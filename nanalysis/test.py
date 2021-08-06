from nanalysis import NumMethods, LinSys, Interpolation, FindRoot
import numpy as np


a = np.array([[10,5,0,0],[5,10,-4,0],[0,-4,8,-1],[0,0,-1,5]])
b = np.array([6,25,-11,-11])



baa = LinSys()
print(baa.sor(a, b, w=1.1, tol=10**(-8), n_0=2))
# (-0.71885, 2.818822, -0.2809726, -2.235422) after 2 iterations

from nanalysis import NumMethods, LinSys, Interpolation, FindRoot
import numpy as np


f = lambda x: np.exp(x)
x = [0, 1, 2, 3]
y = [f(t) for t in x]
fpo = NumMethods(f).mid_3diff(x[0])
fpn = NumMethods(f).mid_3diff(x[3])
a = np.array([[10, 5, 0, 0], [5, 10, -4, 0], [0, -4, 8, -1], [0, 0, -1, 5]])
b = np.array([6, 25, -11, -11])

hphp = Interpolation()
print(hphp.cubic_spline(x=x, approx=0.12, y=y, bound='clamp', fpo=fpo, fpn=fpn))

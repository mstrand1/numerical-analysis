import nanalysis
import numpy as np

f = lambda x: x*np.exp(x)
x = [0, 0.1, 0.2, 0.3]
y = [f(i) for i in x]
b = nanalysis.interpolation
print(b.Interpolation(f).dd_lagrange(x, y,0.13))
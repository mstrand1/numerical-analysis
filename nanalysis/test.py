import nanalysis
import numpy as np

f = lambda x: x*np.exp(x)
x = [0, 0.1, 0.2, 0.3]
b = nanalysis.interpolation
print(b.Interpolation(f).cubic_spline(x,approx=.13))
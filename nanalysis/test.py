import nanalysis
import numpy as np

f = lambda x: np.exp(x)
x = [0,1,2,3]
y=[np.exp(0), np.exp(1), np.exp(2), np.exp(3)]
b = nanalysis.interpolation
print(b.Interpolation().spline_coeff(x,y))
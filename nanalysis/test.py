import nanalysis
import numpy as np

f = lambda x: np.exp(x)
x = [1.3,1.6,1.9]
y=[0.6200860,0.4554022,0.2818186]
yp=[-0.5220232,-0.5698959,-0.5811571]
b = nanalysis.interpolation
print(b.Interpolation(f).hdiv_diff(x,y,yp))
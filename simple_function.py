from dolfin import *
import numpy as np
import matplotlib.pylab as plt

mesh = Interval(100, 0, 2*pi)

V = FunctionSpace(mesh, 'CG', 1)
W = FunctionSpace(mesh, 'CG', 1)

fexp = Expression("sin(x[0])")
f = interpolate(fexp, V)

gexp = Expression("sin(x[0]) + 2")
g = interpolate(gexp, V)

p = 2*np.ones(np.shape(f.vector().array()))

f.vector()[:] = f.vector().array() +p

plt.plot(f.vector().array())
plt.plot(g.vector().array())

plt.show()





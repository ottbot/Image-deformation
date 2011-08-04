from dolfin import *

mesh = Interval(100, 0, 2*pi)
V = FunctionSpace(mesh, 'CG', 1)
fexp = Expression("sin(x[0])")
f = interpolate(fexp, V)

a = dot(f,f)

print a



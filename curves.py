# constant distance function

from dolfin import *

import numpy as np
import matplotlib.pylab as plt

def split_array(q):

    x = q.vector().array()

    X = x[0:np.size(x)/2]
    Y = x[np.size(x)/2: np.size(x)]

    return X,Y

def get_sort_order(fun_space):
    vals = interpolate(Expression(('x[0]','x[0]')), fun_space)
    return np.argsort(split_array(vals)[0])




mesh = Interval(100, 0, 2*pi)

N = 1
M = 1

VD = VectorFunctionSpace(mesh, 'DG', N, dim=2)
dVD = VectorFunctionSpace(mesh, 'DG', N - 1, dim=2)
VC = VectorFunctionSpace(mesh, 'CG', M, dim=2)

# Template
qA = Expression(('sin(x[0])','cos(x[0])'))
# Target
qB = Expression(('2*sin(x[0])','2*cos(x[0])'))


v = TestFunction(VC)

# make array of these:

u = TrialFunction(VC)

q = interpolate(qA, VD)
f = interpolate(qB, VD)

alpha_sq = 1

#dq = project(q.dx(0), dVD)
dq = q.dx(0)
j = sqrt(dot(dq,dq))

B = dot(v,f)*dx
a = dot(v,u)*j*dx + (alpha_sq*dot(v.dx(0),u.dx(0))/j)*dx

problem = VariationalProblem(a,B)



u = problem.solve()

s = assemble(energy_norm(a, u))


# same things, but with mass and stiffness matrices

# m = dot(v,u)*j*dx
# l = (alpha_sq*dot(v.dx(0),u.dx(0))/j)*dx

# A = assemble(a)
# b = assemble(B)

# M = assemble(m)
# L = assemble(l)

# u2 = Function(VC)

# solve(M, u2.vector(), b)

#u = u2
#---------------------
q_prev = interpolate(qA, VD)

plt.ion()
plt.figure()
plt.axis('equal')

line, = plt.plot(*split_array(q_prev))
plt.xlim(-3,3)
plt.ylim(-3,3)


dt = 0.01


r = TestFunction(VD)
q = TrialFunction(VD)

# v = Expression(('sin(x[0])','cos(x[0])'))



a = dot(r,q)*dx
L = dot(q_prev,r)*dx -  dt*dot(r,u)*dx

A = assemble(a)

q = Function(VD)   # the unknown at a new time level
T = 1             # total simulation time
t = dt


sorted = get_sort_order(VD)

while t <= T:
    b = assemble(L)
 
    solve(A, q.vector(), b)

    t += dt
    q_prev.assign(q)

    X,Y = split_array(q)
    line.set_data(X[sorted], Y[sorted])
    #plt.gca().relim()
    #plt.gca().autoscale_view()
    plt.draw()









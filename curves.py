# constant distance function

from dolfin import *

import numpy as np
import matplotlib.pylab as plt

def split_array(q):

    x = q.vector().array()

    X = x[0:np.size(x)/2]
    Y = x[np.size(x)/2: np.size(x)]

    return X,Y



plt.ion()
plt.figure()
#plt.axis('equal')


dt = 0.01


mesh = Interval(100, 0, 2*pi)

# what's happening at higher orders?
V = VectorFunctionSpace(mesh, 'DG', 1, dim=2)

q0 = Expression(('cos(x[0])','sin(x[0])'))


q_prev = interpolate(q0, V)

line, = plt.plot(*split_array(q_prev))
plt.xlim(-3,3)
plt.ylim(-3,3)

r = TestFunction(V)
q = TrialFunction(V)

v = Expression(('.5*x[0]*sin(x[0])','2*cos(x[0])'))


a = dot(r,q)*dx
L = dot(q_prev,r)*dx -  dt*dot(r,v)*dx

A = assemble(a)

q = Function(V)   # the unknown at a new time level
T = 1             # total simulation time
t = dt


while t <= T:
    b = assemble(L)
 
    solve(A, q.vector(), b)

    t += dt
    q_prev.assign(q)

    line.set_data(split_array(q))
    #plt.gca().relim()
    #plt.gca().autoscale_view()
    plt.draw()



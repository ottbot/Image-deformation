import os, sys
lib_path = os.path.abspath('../')
sys.path.append(lib_path)

import numpy as np
import matplotlib 
matplotlib.use("cairo")
import matplotlib.pylab as plt
import image_deformation as img

""" Makes a nice quiver plot to complate inner vs outer metrics """

q = np.genfromtxt("misc/hand/qa.txt", unpack=True)

im = img.Immersion(M=100,N=1, qA=q,qB=q)

W = 600

N = 21
p = W/float(N-1)


U = np.zeros((N,N))
X = 1.0*U
V = 1.0*U
Y = 1.0*U 

f = lambda x: 3*(x - W/2.) + 2*np.sin(x)
g = lambda y: -(y - W/2.)
#f = lambda x: np.sin(x/175. + 100)

for i in xrange(0,N):
    for j in xrange(0,N):
        U[i,j] = f(p*i)
        V[i,j] = g(p*j)
        X[i,j] = p*i
        Y[i,j] = p*j



im.new_figure()
#plt.figure()
#plt.axis("equal")
plt.quiver(X,Y,U,V,color='k',scale_units='inches', scale=3000)
qx,qy = im.split_array(q)
plt.plot(qx,qy,color="k", lw=3)
plt.axis((0,W,0,W))
plt.savefig("misc/outer_field.pdf",bbox_inches='tight')

#no build the quiver for the inner metric plot
u = [f(x) for x in qx]
v = [g(y) for y in qy]

ut = []
vt = []
xt = []
yt = []

for i in np.arange(0,201,2):
    ut.append(u[i])
    vt.append(v[i])
    xt.append(qx[i])
    yt.append(qy[i])


im.new_figure()
plt.quiver(qx,qy,u,v,color="k",scale_units='inches', scale=2000)
plt.plot(qx,qy,color="black", lw=3)
plt.axis((0,600,0,600))
plt.savefig("misc/inner_field.pdf",bbox_inches='tight')

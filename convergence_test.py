import dolfin as dol
import numpy as np

import scipy as sp

import matplotlib
matplotlib.use('cairo')

import matplotlib.pylab as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from image_deformation import Immersion

N = 10
M = 100

im = Immersion(M,N)
        
u = dol.Expression(('cos(x[0])/10.0','cos(x[0])/10.0'))
#u = dol.Expression(('0.1','0.1'))
u = dol.interpolate(u, im.V)

U = np.zeros(im.mat_shape)

u_scaler = 0.2

for n in xrange(im.N):
    U[:,n] = u_scaler * u.vector().array()
            
S  = im.calc_S(U)
        
dSarr = np.reshape(im.calc_dS(U),im.mat_shape)

vdS = 0

#v = dol.Expression(('cos(x[0])/2.0','cos(x[0])/2.0'))
v = dol.Expression(('pow(x[0],2.0)/100.0','pow(x[0],3)/100.0'))
v = dol.interpolate(v, im.V)


s_vdS = 0
s_v = 0

for dS in im.matrix_to_coeffs(dSarr):
    vdS += dol.assemble(dol.inner(v,dS)*dol.dx)

#print s_vdS * im.dt
vdS *= im.dt

lims = []
Sps = []

eps = 10.**(-sp.arange(20))
        

for ep in eps:
    Up = np.zeros(im.mat_shape)
    for n in xrange(im.N):
        Up[:,n] = u_scaler * u.vector().array() + ep*v.vector().array()

    im = Immersion(M,N)
    Sp = im.calc_S(Up)
    Sps.append(Sp)
    lims.append((Sp - S)/ep)


print "%s %15s %15s" % ("Eps","LHS","RHS")
for n in xrange(len(eps)):
    print "%.0e  %15.6f  %15.6f" % (eps[n], vdS, lims[n])


o1_lims = lims

#plt.figure()
#plt.plot(lims)


print "---------------------------"
print "Second order"


lims = []

for ep in eps:
    Upp = np.zeros(im.mat_shape)
    for n in xrange(im.N):
       Upp[:,n] = u_scaler * u.vector().array() + ep*v.vector().array()

    Upn = np.zeros(im.mat_shape)
    for n in xrange(im.N):
        Upn[:,n] = u_scaler * u.vector().array() - ep*v.vector().array()

    imp = Immersion(M,N)
    imn = Immersion(M,N)
                
    Spp = imp.calc_S(Upp)
    Spn = imn.calc_S(Upn)

    lims.append((Spp - Spn)/(2*ep))


print "%s %15s %15s" % ("Eps","LHS","RHS")
for n in xrange(len(eps)):
    print "%.0e  %15.6f  %15.6f" % (eps[n], vdS, lims[n])


o2_lims = lims

#plt.plot(lims)
#plt.show()


E1 = [abs(abs(l) - abs(vdS)) for l in o1_lims]
E2 = [abs(abs(l) - abs(vdS)) for l in o2_lims]

plt.figure()


plt.loglog(eps,E1, label="$E_1$ (First order)")
plt.loglog(eps,E2, label="$E_2$ (Second order)")
plt.legend()
plt.xlim((10**(-13),1))


verts = [(10**-9,10**-1),
         (10**-7,10**-1),
         (10**-7,10**1),
         (10**-9,10**-1)]


codes = [Path.MOVETO, 
         Path.LINETO, 
         Path.LINETO,
         Path.CLOSEPOLY]

path = Path(verts, codes)

patch = patches.PathPatch(path, facecolor='none', lw=1,edgecolor="grey")

ax = plt.gca()
ax.add_patch(patch)

t1x = 10**(np.log10(verts[0][0]) + np.log10(verts[1][0]/verts[0][0])/2.)
t1y = 10**(np.log10(verts[0][1])-0.2)

t2x = 10**(np.log10(verts[1][0])+0.1)
t2y = 10**(np.log10(verts[1][1]) + np.log10(verts[2][1]/verts[1][1])/2.)

ax.text(t1x,t1y,"1")
ax.text(t2x,t2y,"1")

plt.xlabel("$\epsilon$")
plt.ylabel("Error")
plt.grid(True)

plt.draw()

plt.savefig("code_error.pdf")

from image_deformation import Immersion
import drawcurve

import numpy as np

# need to make "dummy" instance just to get the vector size
# --- Fix this!
c = Immersion(100,100)
N = c.vec_size

print "vec -- ", N
tmpl = drawcurve.get_vec(N, "Draw template")
targ = drawcurve.get_vec(N, "Draw target")
print "a : ", np.shape(tmpl), " b: ", np.shape(targ)


c = Immersion(100, 100, qA=tmpl, qB=targ)

U = c.U_initial()

c.calc_S(U)
c.calc_ds(U)

c.plot_steps


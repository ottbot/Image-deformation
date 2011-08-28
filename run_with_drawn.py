import image_deformation as img
import drawcurve

import numpy as np

# need to make "dummy" instance just to get the vector size
# --- Fix this!


M, N = 100, 10

vs = img.template_size(M, N)

tmpl = drawcurve.get_vec(vs, "Draw template")
targ = drawcurve.get_vec(vs, "Draw target")

opt = img.minimize(M, N, qA=tmpl, qB=targ)

#c = img.Immersion(M, N, targ, tmpl)
#c.calc_S(opt[0])

#c.plot_steps()

im = opt[1]

im.plot_steps()

import os, sys
lib_path = os.path.abspath('../')
sys.path.append(lib_path)

import image_deformation as img
import drawcurve

import numpy as np

""" 
You can draw two curves and then find the deformation between them. 
It's best to run this in ipython to inspect results further, make other plots
"""

M, N = 100, 10

vs = img.template_size(M=M, N=N)

tmpl = drawcurve.get_vec(vs, "Draw template")
targ = drawcurve.get_vec(vs, "Draw target")

opt = img.minimize(M=M, N=N, qA=tmpl, qB=targ)

im = opt[1]

im.plot_steps()

# Unit testing

#import dolfin as dol
import numpy as np
import numpy.testing as npt
import scipy as sp
#import matplotlib.pylab as plt

import unittest

import image_deformation as img
import drawcurve

class TestDrawing(unittest.TestCase):

    def test_draw_template_targe(self):

        M = 100
        N = 10
        vs = img.template_size(M, N)

        tmpl = drawcurve.get_vec(vs, "Draw template")
        #targ = drawcurve.get_vec(vs, "Draw target")

        im = img.Immersion(M, N, tmpl)

        npt.assert_allclose(tmpl, im.qA.vector().array())


if __name__ == '__main__':
    unittest.main()



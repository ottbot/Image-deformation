import sys
import matplotlib

matplotlib.use("cairo")

from image_deformation import *

try:
    case = sys.argv[1]
except IndexError:
    print "please provide a case name"
    sys.exit(2)


im = run_case(case)

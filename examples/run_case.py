import os, sys

import matplotlib

matplotlib.use("cairo")

from casestudy import *

""" This script runs with a case name taken from the command line """

try:
    case = sys.argv[1]
except IndexError:
    print "please provide a case name"
    sys.exit(2)


im = run_case(case)

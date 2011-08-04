from image_deformation import Immersion

im = Immersion(20,100)

m = im.min()

im.plot_steps()

print "Done:", m


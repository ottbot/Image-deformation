import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from scipy.interpolate import interp1d

class Trace:

    def __init__(self, filename, num=100, 
                 title = "Click and trace around the image. Close window to finish"):

        self.num = num

        print filename
        self.im = plt.imread(filename)


        H,W,A = self.im.shape
        self.im_size = {'H':H,'W':W,'A':A}
        
        fig = plt.figure()
        plt.imshow(self.im)#, orgin="lower")

        ax = fig.add_subplot(111)
        ax.set_title(title)
        #plt.axis('equal')
        plt.axis((0, W, 0, H))
        

        self.line, = ax.plot([0],[0])

        self.xs = []
        self.ys = []

        cidpress = fig.canvas.mpl_connect('button_press_event', self.onclick)
        cidrelease = fig.canvas.mpl_connect('button_release_event', self.release)
        cidmotion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.drawing = False


        plt.show()


    def closest_point(self, x,y):
        def is_white(px):
            if px.size is 3:
                return (px.sum() ==  3.0)
            else:
                if px[3]:
                    return (px.sum() ==  3.0)
                else:
                    return True


        def search(px):
            dist = 0
            fuzz = 10

            for i in xrange(int(x - fuzz), int(x + fuzz)):
                if x > 0 and x < self.im_size['W']:
                    tpx = self.im[i,y]
                    if is_white(px):
                        if not is_white(tpx):
                            return i
                        
                    else:
                        if is_white(tpx):
                            return i
            return False


        def right_search(px):
            dist = 0
            rbnd = int(x) + 50
            if self.im_size['W'] < rbnd:
                rbnd = self.im_size['W']

            if is_white(px):
                for i in xrange(int(x), rbnd):
                    tpx = self.im[i,y]
                    if is_white(tpx):
                        dist += 1
                    else:
                        print "right got black!:", dist
                        return dist
                return False
            else:
                for i in xrange(int(x), rbnd):
                    tpx = self.im[i,y]
                    if is_white(tpx):
                        print "right got black!:", dist
                        return dist
                    else:
                        dist += 1
                return False


        def left_search(px):
            dist = 0
            lbnd = int(x) - 50
            if lbnd < 0:
                lbnd = 0

            if is_white(px):
                for i in reversed(xrange(lbnd, int(x))):
                    tpx = self.im[i,y]
                    if is_white(tpx):
                        dist += 1
                    else:
                        print " left - got black!: ", dist
                        return dist
                return False
            else:
                for i in reversed(xrange(lbnd, int(x))):
                    tpx = self.im[i,y]
                    if is_white(tpx):
                        print " left - got black!: ", dist
                        return dist
                    else:
                        dist += 1
                return False


        #h = self.im_size['H'] - 1 - y
        px = self.im[x,y]
        
        ld = left_search(px)
        rd = right_search(px)


        if ld is False and rd is False:
            print "all white!"
            x = False
        else:
            if ld is False:
                print "no left, so must be right "
                x += rd
            elif rd is False:
                print "no right, so must be left"
                x -= ld
            else:
                if ld < rd:
                    x -= ld
                else:
                    x += rd

        # res = search(px)
        # if res:
        #     print "got a RES:", res
        #     x = res
        # else:
        #     x = False

        return x, y

    def onclick(self, event):
        if event.button == 1 and event.ydata and event.ydata:
            self.drawing = True

            x,y = self.closest_point(event.xdata, event.ydata)

            if x:
                self.xs, self.ys = [x], [y]
            else:
                self.xs, self.ys = [], []

            self.line.set_data([self.xs],[self.ys])


    def on_motion(self,event):
        if self.drawing and event.ydata and event.ydata:
            #print 'x=%d, y=%d, xdata=%f, ydata=%f' \
            #    %(event.x, event.y, event.xdata, event.ydata)

            x,y = self.closest_point(event.xdata, event.ydata)

            if x:
                self.xs.append(x)
                self.ys.append(y)

                self.line.set_data(self.xs, self.ys)
                self.line.figure.canvas.draw()

    def release(self,event):
        self.drawing = False

        # smooth
        self.xs = self.smooth_gaussian(self.xs)
        self.ys = self.smooth_gaussian(self.ys)

        # close it up
        self.xs = np.append(self.xs, self.xs[0])
        self.ys = np.append(self.ys, self.ys[0])

        # now interpolate to set number of values
        xs = self.interp(self.ys, self.xs)
        ys = self.interp(self.xs, self.ys)

        self.xs = xs
        self.ys = ys

        self.line.set_data(self.xs,self.ys)
        self.line.figure.canvas.draw()

    def interp(self, x, y):
        xrang = np.linspace(x.min(), x.max(), np.size(x))
        f = interp1d(xrang, y)
        xrang = np.linspace(x.min(), x.max(), self.num)

        return f(xrang)


    #http://www.swharden.com/blog/2008-11-17-linear-data-smoothing-in-python/
    def smooth_gaussian(self, arr, degree=5):  
         window=degree*2-1  
         weight=np.array([1.0]*window)  
         weightGauss=[]  
         for i in range(window):  
             i=i-degree+1  
             frac=i/float(window)  
             gauss=1/(np.exp((4*(frac))**2))  
             weightGauss.append(gauss)  
         weight=np.array(weightGauss)*weight  
         smoothed=[0.0]*(len(arr)-window)  
         for i in range(len(smoothed)):  
             smoothed[i]=sum(np.array(arr[i:i+window])*weight)/sum(weight)  
         return smoothed 



# ----------------------
def get_vec(filename="cat.png", num=100, title = None):
    x,y = get_xy(filename, num, title)
    return np.append(c.xs, c.ys)

def get_xy(filename="cat.png", num=100, title = None):
    c = Trace(filename, num/2., title)
    return c.xs, c.ys

#run.. REMOVE THIS
x, y = get_xy("ast500.png")

plt.plot(x,y)

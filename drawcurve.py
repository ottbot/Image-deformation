import numpy as np
import matplotlib.pylab as plt

from scipy.interpolate import interp1d

# example: DrawCurve(100, "Draw template")
# the first argument is the size of the interpolated vector
class DrawCurve:
    
    def __init__(self, num = 100, title = "Click and hold to draw"):
        #self.smoothing = "hanning"
        self.num = num
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        plt.axis('equal')
        
        self.line, = ax.plot([0],[0])

        cidpress = fig.canvas.mpl_connect('button_press_event', self.onclick)
        cidrelease = fig.canvas.mpl_connect('button_release_event', self.release)
        cidmotion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.drawing = False

        plt.show()

    def onclick(self, event):
        if event.button == 1 and event.ydata and event.ydata:
            self.drawing = True

            self.xs, self.ys = [event.xdata], [event.ydata]
            self.line.set_data([self.xs],[self.ys])


    def on_motion(self,event):
        if self.drawing and event.ydata and event.ydata:
            #print 'x=%d, y=%d, xdata=%f, ydata=%f' \
            #    %(event.x, event.y, event.xdata, event.ydata)

            self.xs.append(event.xdata)
            self.ys.append(event.ydata)

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



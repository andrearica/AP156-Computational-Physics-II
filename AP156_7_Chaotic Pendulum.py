# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:38:57 2018

@author: Andrea Rica
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.animation as animation
from scipy.integrate import odeint


def f(r,t):
    phi = r[0]
    omega = r[1]
    fx = omega
    fy = -ar*omega - np.sin(phi) + a*np.cos(wd*t)
    return np.array([fx,fy],float)
    
def RK42(f,ts,r):
    xs = []
    ys = []
    pcsx = []
    pcsy = []
    for t in ts:
        if any(ts1==t):
            pcsx.append(r[0])
            pcsy.append(r[1])
        xs.append(r[0])
        ys.append(r[1])
        k1 = h*f(r,t)
        k2 = h*f(r+(1/2*k1),t+(1/2*h))
        k3 = h*f(r+(1/2*k2),t+(1/2*h))
        k4 = h*f(r+(k3),t+h)
        r += (k1 +2*k2 +2*k3 +k4)/6
    
    return xs,ys,ts, pcsx, pcsy
    #plt.plot (ts,ys)
    
#initial values
wd = 2/3
a = 0.85
ar = 0.25
p = (np.pi/2)
w = 0
r = np.array([p, w],float)

#time array
a1 = 0
a2 = 10000*np.pi
h = (a2-a1)/1000000
ts = np.arange(a1,a2,h)

#poincare
#j = np.arange(0,100000)
#ts1 = 2*np.pi*j/wd
#tnew = ts1*j
a3 = 0
a4 = 100000*np.pi
h1 = 3*np.pi
tnew = np.arange(a3,a4,h1)

#integrate
fun = odeint(f,r,ts)
#fun = odeint(f,r,tnew)
#fun = RK42(f,ts,r)


#plt.scatter(fun[:,0],fun[:,1],s=0.5)
#plt.savefig("Poincare 1p3")
plt.plot(fun[:,0],fun[:,1])
#plt.xlim(-np.pi,np.pi)
#plt.ylim(-np.pi,np.pi)
#plt.show()

#Animation
#fig,ax = plt.subplots()  
#    
#y = 0
#line, = ax.plot(fun[:y,0], fun[:y,1])
#
#
#def animate(i):
#    line.set_xdata(fun[:y+i,0])
#    line.set_ydata(fun[:y+i,1]) # update the data.
#    plt.xlabel("t = " str(ts[i]/np.pi)+ r'$\pi$')
#    return line,
#
#
#animm = animation.FuncAnimation(fig, animate, interval=0.01)
#
#
#    
plt.xlim(-np.pi,np.pi)
plt.ylim(-np.pi,np.pi)
plt.xlim(min(fun[:,0]),max(fun[:,0]))
plt.ylim(min(fun[:,1]),max(fun[:,1]))
plt.show()


#plt.scatter(fun[3],fun[4],color = 'red')
#plt.show()
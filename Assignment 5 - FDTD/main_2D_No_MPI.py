# -*- coding: utf-8 -*-
"""
Late edited: March 2022

@author: Andrew Hayman
"""

"""
FDTD simulation for 2D. The program allows the user to specify a space grid
dimension and number of time steps. Internally, the code generates a gaussian
pulse at the center of the grid, and inserts a dielectric box to the top right
of the pulse. Different methods can be used, which will all give the same 
outputs, but with different timing performance. 

You can optionally play an animate, or save snapshots at any desried time 
steps. 

Note that all methods use the same structure for consistency.   

Instructions: 
1. If you want animations, be sure to set %matplotlib auto in the console
2. Scroll down to questions section and call the function with any desired 
method. 

"""
#%% Imports
import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.constants as constants
import matplotlib as mpl
import timeit
import numba 

#%% Matplotlib settings
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5

#%% Constants
femto = constants.femto
tera = constants.tera
micro = constants.micro
nano = constants.nano
peta = constants.peta

#%% Main Simulation 
def simulate_fdtd(Xmax=250, 
              Ymax = 250, 
              time_steps=500, 
              method="numba", 
              animate=True, 
              snapshots=None): 
    
    # Constants
    c = constants.c
    dxy = 20*nano
    dt = dxy/(2.*constants.c)
    frame_freq = 100

    # Initialize field arrays and permittivity arrays
    Dz = np.zeros((Xmax, Ymax), float)  
    Ez = np.zeros((Xmax, Ymax), float)  
    Hx = np.zeros((Xmax, Ymax), float)
    Hy = np.zeros((Xmax, Ymax), float)
    ga=np.ones((Xmax, Ymax),float)
    
    if(animate or snapshots): 
        EzMonTime1=[]
        PulseMonTime=[]
    
    # Dipole source position at center
    isource = int(Ymax/2)
    jsource = int(Xmax/2)
    
    # Pulse definition
    spread=2.* femto/dt
    t0=spread*6
    freq_in = 2*np.pi*200*tera
    w_scale = freq_in*dt
    lam = 2*np.pi*c/freq_in
    
    # Dielectric box. Fill in ga with epsilon value
    eps = 9
    X1=isource+10
    X2=X1+40
    Y1=jsource+10
    Y2=Y1+40
    for j in range (0,Ymax): 
        for i in range (0,Xmax):
            if i>X1 and i<X2+1 and j>Y1 and j<X2+1:   
                ga[i,j] = 1./eps
                
    # Prepare figure 
    if animate or snapshots: 
        fig = plt.figure(figsize=(3.2,3.2))
        x_vals = np.arange(0,Xmax, 1)
        y_vals = np.arange(0,Ymax,1)
        x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
        def update_plot(): 
            plt.clf()
            ax = fig.add_axes([.25, .25, .6, .6])  
            ax2 = fig.add_axes([0, .7, .15, .15])  
            img = ax.contourf(x_mesh,y_mesh,Ez)
            cbar=plt.colorbar(img, ax=ax)
            cbar.set_label('$Ez$ (arb. units)')
            
            ax.vlines(X1,Y1,Y2,colors='r')
            ax.vlines(X2,Y1,Y2,colors='r')
            ax.hlines(Y1,X1,X2,colors='r')
            ax.hlines(Y2,X1,X2,colors='r')
            
            ax.set_xlabel('Grid Cells ($x$)')
            ax.set_ylabel('Grid Cells ($y$)')
            
            ax.set_title("Frame " + str(t))
            
            PulseNorm = np.asarray(PulseMonTime)*0.2;
            ax2.plot(PulseNorm,'r',linewidth=1.6)
            ax2.plot(EzMonTime1,'b',linewidth=1.6)
            ax2.set_yticklabels([])
            ax2.set_xticklabels([])
            ax2.set_title('$E_{center}(t),P_{in}$')
            
            if t in snapshots: 
                plt.savefig("snapshot_" + str(t) + ".pdf", format='pdf', dpi=1200,bbox_inches = 'tight')
                
            if animate: 
                plt.show() 
                plt.pause(0.01)
        
    if method=="loop_row_major":
        def update_H(Hx, Hy, Ez): 
            for x in range (0,Xmax-1): 
                for y in range (0,Ymax-1): 
                    Hx[x,y] = Hx[x,y]+ 0.5*(Ez[x,y]-Ez[x,y+1])                       
                    Hy[x,y] = Hy[x,y]+ 0.5*(Ez[x+1,y]-Ez[x,y]) 
            return Hx, Hy
        
        def update_E(Hx, Hy, Dz, Ez, ga): 
            for x in range (1,Xmax-1): 
                for y in range (1,Ymax-1):
                    Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1]) 
                    Ez[x,y] =  ga[x,y]*(Dz[x,y])
            return Ez, Dz  
     
    if method=="loop_col_major":
        def update_H(Hx, Hy, Ez): 
            for y in range (0,Ymax-1): 
                for x in range (0,Xmax-1): 
                    Hx[x,y] = Hx[x,y]+ 0.5*(Ez[x,y]-Ez[x,y+1])                       
                    Hy[x,y] = Hy[x,y]+ 0.5*(Ez[x+1,y]-Ez[x,y])   
            return Hx, Hy
        
        def update_E(Hx, Hy, Dz, Ez, ga): 
            for y in range (1,Ymax-1):
                for x in range (1,Xmax-1): 
                    Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1]) 
                    Ez[x,y] =  ga[x,y]*(Dz[x,y])
            return Ez, Dz  
                
    if method=="vectorized":
        
        def update_H(Hx, Hy, Ez): 
            Hx[0:(Xmax-1),0:(Ymax-1)] += 0.5*(Ez[0:(Xmax-1),0:(Ymax-1)]-Ez[0:(Xmax-1),1:Ymax])                       
            Hy[0:(Xmax-1),0:(Ymax-1)] += 0.5*(Ez[1:Xmax,0:(Ymax-1)]-Ez[0:(Xmax-1),0:(Ymax-1)]) 
            return Hx, Hy
        
        def update_E(Hx, Hy, Dz, Ez, ga): 
            Dz[isource,jsource] += np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt)) 
            Dz[1:(Xmax-1),1:(Ymax-1)] += 0.5*(Hy[1:(Xmax-1),1:(Ymax-1)]
                                              -Hy[:(Xmax-2),1:(Ymax-1)]
                                              -Hx[1:(Xmax-1),1:(Ymax-1)]
                                              +Hx[1:(Xmax-1),:(Ymax-2)]) 
            Ez[1:(Xmax-1),1:(Ymax-1)] =  ga[1:(Xmax-1),1:(Ymax-1)]*(Dz[1:(Xmax-1),1:(Ymax-1)])
            return Ez, Dz  
                
    if method=="numba_loop_row_major":
        @numba.jit(nopython=True, fastmath=True)
        def update_H(Hx, Hy, Ez): 
            for x in range(0,Xmax-1): 
                for y in range(0,Ymax-1): 
                    Hx[x,y] = Hx[x,y]+ 0.5*(Ez[x,y]-Ez[x,y+1])                       
                    Hy[x,y] = Hy[x,y]+ 0.5*(Ez[x+1,y]-Ez[x,y]) 
            return Hx, Hy
        
        @numba.jit(nopython=True, fastmath=True)
        def update_E(Hx, Hy, Dz, Ez, ga): 
            for x in range (1,Xmax-1):
                for y in range (1,Ymax-1): 
                    Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1]) 
                    Ez[x,y] =  ga[x,y]*(Dz[x,y])
            return Ez, Dz  
           
    if method=="numba_loop_col_major":
        @numba.jit(nopython=True, fastmath=True)
        def update_H(Hx, Hy, Ez): 
            for y in range(0,Ymax-1): 
                for x in range(0,Xmax-1): 
                    Hx[x,y] = Hx[x,y]+ 0.5*(Ez[x,y]-Ez[x,y+1])                       
                    Hy[x,y] = Hy[x,y]+ 0.5*(Ez[x+1,y]-Ez[x,y]) 
            return Hx, Hy
        
        @numba.jit(nopython=True, fastmath=True)
        def update_E(Hx, Hy, Dz, Ez, ga): 
            for y in range (1,Ymax-1):
                for x in range (1,Xmax-1): 
                    Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1]) 
                    Ez[x,y] =  ga[x,y]*(Dz[x,y])
            return Ez, Dz  
                
    if method=="numba_vectorized":
        @numba.jit(nopython=True, fastmath=True)
        def update_H(Hx, Hy, Ez): 
            Hx[0:(Xmax-1),0:(Ymax-1)] += 0.5*(Ez[0:(Xmax-1),0:(Ymax-1)]-Ez[0:(Xmax-1),1:Ymax])                       
            Hy[0:(Xmax-1),0:(Ymax-1)] += 0.5*(Ez[1:Xmax,0:(Ymax-1)]-Ez[0:(Xmax-1),0:(Ymax-1)]) 
            return Hx, Hy
        
        @numba.jit(nopython=True, fastmath=True)
        def update_E(Hx, Hy, Dz, Ez, ga): 
            Dz[1:(Xmax-1),1:(Ymax-1)] += 0.5*(Hy[1:(Xmax-1),1:(Ymax-1)]
                                              -Hy[:(Xmax-2),1:(Ymax-1)]
                                              -Hx[1:(Xmax-1),1:(Ymax-1)]
                                              +Hx[1:(Xmax-1),:(Ymax-2)]) 
            Ez[1:(Xmax-1),1:(Ymax-1)] =  ga[1:(Xmax-1),1:(Ymax-1)]*(Dz[1:(Xmax-1),1:(Ymax-1)])
            return Ez, Dz  

    if method=="fast":
        @numba.jit(nopython=True, fastmath=True, cache=True)
        def update_E(Hx, Hy, Dz, Ez, ga): 
            for x in range (1,Xmax-1): 
                for y in range (1,Ymax-1):
                    Dz[x,y] += 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1]) 
                    Ez[x,y] = Dz[x,y]
                    
            for x in range(X1+1, X2+1): 
                for y in range(Y1+1, Y2+1): 
                    Ez[x,y] /= eps            
            return Ez, Dz  
        
        @numba.jit(nopython=True, fastmath=True, cache=True)
        def update_H(Hx, Hy, Ez): 
            for x in range(0,Xmax-1): 
                for y in range(0,Ymax-1): 
                    Hx[x,y] += 0.5*(Ez[x,y]-Ez[x,y+1])                  
                    Hy[x,y] += 0.5*(Ez[x+1,y]-Ez[x,y])
            return Hx, Hy
         
    # Main loop
    start = timeit.default_timer() 
    for t in range (0,time_steps):
        if(animate or snapshots): 
            EzMonTime1.append(Ez[isource,jsource]) 
            PulseMonTime.append(np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt))) 
        Dz[isource,jsource] += np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt)) 
        Ez, Dz = update_E(Hx, Hy, Dz, Ez, ga) 
        Hx, Hy = update_H(Hx, Hy, Ez)                               
        if (animate and t%frame_freq==0) or t in snapshots: 
            update_plot()
    end = timeit.default_timer()  
    return (end-start)
                
#%% Q3a), Q3b), Q3c) 
# 7 Methods listed here for conveniant
fast_methods = ["vectorized", 
           "numba_loop_row_major", 
           "numba_loop_col_major", 
           "numba_vectorized", 
           "fast"]

slow_methods = ["loop_row_major", 
           "loop_col_major"]


# 500x500 grid for 1000 time steps
for method in fast_methods: 
    time = simulate_fdtd(Xmax=500, 
                  Ymax=500, 
                  time_steps=1000, 
                  method=method, 
                  animate=False, 
                  snapshots=[])
    print("method: ", method, " time: ", time)
    
for method in slow_methods: 
    time = simulate_fdtd(Xmax=500, 
                  Ymax=500, 
                  time_steps=10, 
                  method=method, 
                  animate=False, 
                  snapshots=[])
    print("method: ", method, " time: ", 100*time)
    
# 1000x1000 grid for 1000 time steps
for method in fast_methods: 
    time = simulate_fdtd(Xmax=1000, 
                  Ymax=1000, 
                  time_steps=1000, 
                  method=method, 
                  animate=False, 
                  snapshots=[])
    print("method: ", method, " time: ", time)
    
for method in slow_methods: 
    time = simulate_fdtd(Xmax=1000, 
                  Ymax=1000, 
                  time_steps=10, 
                  method=method, 
                  animate=False, 
                  snapshots=[])
    print("method: ", method, " time: ", 100*time)
# -*- coding: utf-8 -*-
"""
Late edited: March 2022

@author: Andrew Hayman
"""

"""
Same as main_2D_No_MPI.py but now with MPI implementing the fastest method. 
The simulation is split up over the x-axis and buffers are added to receive
the missing Ez and Hy parts from adjacent processes. The source point and 
box are all correctly distributed across nodes.

Instructions: 
1. Change main function call to any desired grid size & number of step steps. 
Set any snapshots you want to save. 
2. Run the main.sh bash script to time the code for 1, 2, 4, 8, 16 parallel
processes.

"""
#%% Imports
import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.constants as constants
import matplotlib as mpl
import timeit
import numba 
from mpi4py import MPI

#%% Maplotlib settings
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

#%% Setup MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#%% Main Simulation 
def simulate_fdtd(Xmax,Ymax,time_steps,snapshots=None): 

    # Constants
    c = constants.c
    dxy = 20*nano
    dt = dxy/(2.*constants.c)
    frame_freq = 100
    
    # Get chunk size for MPI and get indices for loops and gathering
    # Note that dH is used to offset the Hy indexing due to the buffer at the 
    # start of Hy
    chunk = int(Xmax / size)

    E_start = 0
    E_stop = chunk
    H_start = 0
    H_stop = chunk
    Ez_idx = -1
    dH = 1
    
    if rank==0:
        E_start = 1
        dH = 0
        
    if rank==(size-1):
        E_stop = chunk-1
        H_stop = chunk-1
        Ez_idx = None
    
    # Initialize arrays 
    Dz = np.zeros((chunk, Ymax), float)  
    Hx = np.zeros((chunk, Ymax), float)
    if rank==0:
        Hy = np.zeros((chunk, Ymax), float)
    else: 
        Hy = np.zeros((chunk+1, Ymax), float)
    if rank==(size-1):
        Ez = np.zeros((chunk, Ymax), float)
    else: 
        Ez = np.zeros((chunk+1, Ymax), float)
    
    # Dipole source position at center
    jsource = int(Ymax/2)
    isource = int(Xmax/2)
    isource_idx=isource%chunk
    if size%2==1: 
        pulse_rank = int((size-1)/2)
    else: 
        pulse_rank = int(size/2)
    
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
                
    # Get the right indices for the dielectric box
    cond_A = X1>(rank*chunk) and X1<((rank+1)*chunk)
    cond_B = X2>(rank*chunk) and X2<((rank+1)*chunk)
    cond_C = X1<(rank*chunk) and X2>=((rank+1)*chunk)
    
    if cond_C: 
        X1_idx = 0 
        X2_idx = chunk
    elif cond_A and cond_B: 
        X1_idx = X1%chunk 
        X2_idx = X2%chunk 
    elif cond_A:
        X1_idx = X1%chunk
        X2_idx = chunk 
    elif cond_B: 
        X1_idx = 0
        X2_idx = X2%chunk 
    else:
        X1_idx = chunk+1
        X2_idx = chunk 
    
    if snapshots and rank==0: 
        fig = plt.figure(figsize=(3.2, 3.2))
        x_vals = np.arange(0,Xmax, 1)
        y_vals = np.arange(0,Ymax,1)
        x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
        def update_plot(Ez_full, t): 
            plt.clf()
            ax = fig.add_axes([.25, .25, .6, .6])  
            img = ax.contourf(x_mesh,y_mesh,Ez_full)
            cbar=plt.colorbar(img, ax=ax)
            cbar.set_label('$Ez$ (arb. units)')
            
            ax.vlines(X1,Y1,Y2,colors='r')
            ax.vlines(X2,Y1,Y2,colors='r')
            ax.hlines(Y1,X1,X2,colors='r')
            ax.hlines(Y2,X1,X2,colors='r')
            
            ax.set_xlabel('Grid Cells ($x$)')
            ax.set_ylabel('Grid Cells ($y$)')
            
            ax.set_title("Frame " + str(t))
            
            if t in snapshots: 
                plt.savefig("mpi_snapshot_" + str(t) + ".pdf", format='pdf', dpi=1200,bbox_inches = 'tight')
        
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def update_E(Hx, Hy, Dz, Ez): 
        for x in range (E_start,E_stop): 
            for y in range (1,Ymax-1):
                Dz[x,y] += 0.5*(Hy[x+dH,y]-Hy[x+dH-1,y]-Hx[x,y]+Hx[x,y-1])
                Ez[x,y] = Dz[x,y]
                
                if(x>X1_idx and x<(X2_idx+1) and y>Y1 and y<(Y2+1)): 
                    Ez[x,y] /= eps   
        return Ez, Dz  
    
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def update_H(Hx, Hy, Ez): 
        for x in range(H_start,H_stop): 
            for y in range(0,Ymax-1):   
                Hx[x,y] += 0.5*(Ez[x,y]-Ez[x,y+1])                  
                Hy[x+dH,y] += 0.5*(Ez[x+1,y]-Ez[x,y])
        return Hx, Hy
           
         
    start = timeit.default_timer() 
    for t in range (0,time_steps):
        
        # Insert Pulse
        if rank==pulse_rank:
            Dz[isource_idx,jsource] += np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt))  
              
        # Update Ez, Dz
        Ez, Dz  = update_E(Hx, Hy, Dz, Ez)  

        # Get Ez[x+1]        
        if rank!=0:
            comm.send(Ez[0,:], dest=rank - 1, tag=11)
        if rank!=size-1:
            Ez[-1,:] = comm.recv(source=rank + 1, tag=11)
        
        # Update Hx, Hy
        Hx, Hy = update_H(Hx, Hy, Ez)
        
        # Get Hy[x-1]
        if rank!=size-1:
            comm.send(Hy[-1,:], dest=rank + 1, tag=12)
        if rank!=0:
            Hy[0,:] = comm.recv(source=rank - 1, tag=12)
                   
        # Gather to save snapshot if applicable
        if t in snapshots:                
            Ez_full = None             
            if rank==0: 
                Ez_full = np.zeros((Xmax, Ymax), float)  
            comm.Gather(Ez[:Ez_idx,:], Ez_full, root=0)

            if rank==0:
                    update_plot(Ez_full, t)
                
    end = timeit.default_timer()  
    print(end-start)
               
#%% Q3d)
simulate_fdtd(Xmax=3008, 
              Ymax=3008, 
              time_steps=1000, 
              snapshots=[])
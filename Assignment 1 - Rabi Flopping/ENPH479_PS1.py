# -*- coding: utf-8 -*-
"""
ENPH479 Extreme Nonlinear Optics: Dynamics of Couple ODEs
Last Modified: Jan 22 2022
@author: Andrew Hayman

This program examines Rabi flopping from Gaussian pulses with the RWA and 
without the RWA. First, a continuous pulse under the RWA is examined. Then, 
a time-normalized Gaussian pulse under the RWA is examined, along with 
experiments varying the detuning and dephasing. Finally, a time-normalized
Gaussian pulse is applied without the RWA under different system frequencies
to illustrate the breakdown of the area thereom. 

The program is setup to be modular and easily adaptible to new differential
equation solver methods and new laser pulses. A main experiment routine has
been setup to minimize the repetition of code. 

Many methods make use of args so that the relevant function 
parameters can be passed throughe easily.
"""
#%%
"Imports"
from scipy.integrate import odeint
import matplotlib as mpl
import matplotlib.pyplot as plt  
import math as m  
import numpy as np 
import timeit 

#%%
"Set Matplotlib Defaults"
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5
    
#%% 
"Differential Equation Solvers"

"Simple Euler ODE Solver"
def EulerForward(f,y,t,h, *args):
    k1 = h*np.asarray(f(y,t, *args))                     
    y=y+k1
    return y 

"Runge-Kutta 4 ODE Solver"
def RK4(f,y,t,h, *args): 
    k1 = f(y,t, *args)
    k2 = f(y+h*k1/2, t+h/2, *args)
    k3 = f(y+h*k2/2, t+h/2, *args)
    k4 = f(y+h*k3, t+h, *args)
    y = y + (1/6)*h*(k1+2*k2+2*k3+k4)
    return y

#%%
"OBE Equations for RWA and non-RWA"

"OBE in RWA"
def rwa(y, t, pulse, Omega, Detuning, Dephasing): 
    Omega_t = pulse(Omega, t)

    dy=np.zeros((len(y))) 
    
    dy[0] = -Dephasing*y[0] + Detuning*y[1]
    dy[1] = -Dephasing*y[1] - Detuning*y[0] + Omega_t/2*(2.*y[2]-1.)
    dy[2] = -Omega_t*y[1]
    return dy

"OBE in non-RWA"
def non_rwa(y,t, pulse, Omega, omega, Dephasing, phi):
    Omega_t = pulse(Omega, t)
    Omega_t_sin = Omega_t*np.sin(omega*(t-5) + phi)
    
    dy=np.zeros((len(y))) 

    dy[0] = -Dephasing*y[0] + omega*y[1]
    dy[1] = -Dephasing*y[1] - omega*y[0] + Omega_t_sin*(2.*y[2]-1.)
    dy[2] = -2*Omega_t_sin*y[1]
    return dy


#%% 
"Pulse equations"

"Continuous wave pulse"
def continuous_wave(Omega, t): 
    return Omega

"Time-normalized gaussian pulse centered at t=5"
def gaussian(Omega, t):
    return Omega*np.exp(-(t-5)**2)

#%% 
"Experiments"

"Numerical OBE"
def OBE(method, derivative, tlist, dt, *args): 
    npts = len(tlist)
    y=np.zeros((npts,3))
    yinit = np.array([0.0,0.0,0.0])
    y1=yinit
    y[0,:]= y1
    
    for i in range(1,npts):  
        y1=method(derivative,y1,tlist[i-1],dt, *args) 
        y[i,:]= y1
    
    return y

"Analytical OBE"
def Analytical_OBE(tlist, dt, Omega):
    npts = len(tlist)
    yexact = [m.sin(Omega*tlist[i]/2)**2 for i in range(npts)]
    return yexact

#%% 
"Plotting functions"

"For vertically stacked plots"
def multi_plot(x, y, subtitles, xlabel, ylabel, savename, xlabel_pos=(0.54, -0.02), ylabel_pos=(-0.02, 0.5), fig_size=(3.2, 3.5), fig_padding=0.3, legend_loc=(0.1, 1.00)): 

    # Create sub-plots
    num_plots = len(y)
    fig, axes = plt.subplots(num_plots,figsize=fig_size)
    
    # Titles
    fig.text(xlabel_pos[0], xlabel_pos[1], xlabel, ha='center', linespacing=1.5)
    fig.text(ylabel_pos[0], ylabel_pos[1], ylabel, va='center', ha='center', rotation='vertical', linespacing=1.5)

    # Loop through plots
    colors = ['r', 'b', 'g', 'c', 'm']
    for i in range(num_plots):
        axes[i].plot(x, y[i], colors[i], linewidth=0.75)
        axes[i].xaxis.set_tick_params(which='major', size=3, width=0.5, direction='in', right='on')
        axes[i].yaxis.set_tick_params(which='major', size=3, width=0.5, direction='in', right='on')
        axes[i].grid(True, linestyle=':')
    
    fig.legend(labels=subtitles, loc="lower left", ncol=3, bbox_to_anchor=legend_loc, borderaxespad=0.)
    fig.tight_layout(pad=fig_padding)
    
    plt.savefig(savename+'.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.show()
    
#%%
"For horizontally stacked plots"
def multi_plot_2(x, y, xlabels, ylabel, savename, ylabel_pos=(-0.02, 0.5), fig_size=(3.2, 3.5)): 

    # Create sub-plots
    num_plots = len(y)
    fig, axes = plt.subplots(1,num_plots,figsize=fig_size)
    fig.text(ylabel_pos[0], ylabel_pos[1], ylabel, va='center', ha='center', rotation='vertical', linespacing=1.5)
    
    # Loop through plots
    colors = ['r', 'b', 'g', 'c', 'm']
    for i in range(num_plots):
        axes[i].plot(x[i], y[i], colors[i], linewidth=0.75)

        axes[i].xaxis.set_tick_params(which='major', size=3, width=0.5, direction='in', right='on')
        axes[i].yaxis.set_tick_params(which='major', size=3, width=0.5, direction='in', right='on')
        axes[i].grid(True, linestyle=':')
        axes[i].set_xlabel(xlabels[i])
    
    fig.tight_layout(pad=0.3)
    plt.savefig(savename+'.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.show()
    
#%%
"For grid plots"
def multi_plot_3(x, y, subtitles, xlabels, ylabels, savename, fig_size=(10, 3.5), legend_loc=(0.1, 1.00), y_title_space=(-0.02,0.35)): 

    # Create sub-plots
    num_plots_y = len(y[0])
    num_plots_x = len(y)
    fig, axes = plt.subplots(num_plots_y,num_plots_x,figsize=fig_size)
    
    # Create settings for each subplot
    colors = ['r', 'b', 'g', 'c', 'm']
    
    # Loop through plots
    for i in range(num_plots_x):
        for j in range(num_plots_y): 
            if(i==0): 
                axes[j, i].plot(x, y[i][j], colors[j], linewidth=0.75)
            else:
                axes[j, i].plot(x, y[i][j], colors[j], linewidth=0.75, label='_nolegend_')
    
            axes[j, i].xaxis.set_tick_params(which='major', size=3, width=0.5, direction='in', right='on')
            axes[j, i].yaxis.set_tick_params(which='major', size=3, width=0.5, direction='in', right='on')
            axes[j, i].grid(True, linestyle=':')
            
            # Subtitles
            if(j==(num_plots_y-1)):
                axes[j, i].set_xlabel(xlabels[i])
                
            if(num_plots_y%2==1 and j==((num_plots_y-1)/2)): 
                axes[j, i].set_ylabel(ylabels[i])
    
    if(num_plots_y%2==0):
        for i in range(num_plots_x): 
            fig.text(i*y_title_space[1]-y_title_space[0], 0.55, ylabels[i], va='center', ha='center', rotation='vertical', linespacing=1.5)
        
    fig.legend(labels=subtitles, loc="lower left", ncol=5, bbox_to_anchor=legend_loc, borderaxespad=0.)
    
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(wspace=0.5)
    
    plt.savefig(savename+'.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.show()
    
#%%
"For single plot"
def single_plot(x, y, xlabel, ylabel, savename, xlabel_pos=(0.52, -0.08), ylabel_pos=(0.0, 0.5), fig_size=(3.2, 1.5)): 

    # Create plot
    fig, axes = plt.subplots(figsize=fig_size)
    
    # Titles
    fig.text(xlabel_pos[0], xlabel_pos[1], xlabel, ha='center', linespacing=1.5)
    fig.text(ylabel_pos[0], ylabel_pos[1], ylabel, va='center', ha='center', rotation='vertical', linespacing=1.5)
     
    # Create settings
    axes.plot(x, y, 'r', linewidth=1.0)
    axes.xaxis.set_tick_params(which='major', size=3, width=0.5, direction='in', right='on')
    axes.yaxis.set_tick_params(which='major', size=3, width=0.5, direction='in', right='on')
    axes.grid(True, linestyle=':')
    
    plt.savefig(savename+'.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.show()
#%%
"For semilog plot"
def multi_semilogy(x, y, subtitles, xlabel, ylabel, savename, xlabel_pos=(0.54, -0.02), ylabel_pos=(-0.02, 0.5), fig_size=(3.2, 3.5), fig_padding=0.3, legend_loc=(0.1, 1.00)): 

    # Create sub-plots
    num_plots = len(y)
    fig, axes = plt.subplots(num_plots,figsize=fig_size, sharey=True)
    
    # Titles
    fig.text(xlabel_pos[0], xlabel_pos[1], xlabel, ha='center', linespacing=1.5)
    fig.text(ylabel_pos[0], ylabel_pos[1], ylabel, va='center', ha='center', rotation='vertical', linespacing=1.5)
    
    # Loop through plots
    colors = ['r', 'b', 'g', 'c', 'm']
    for i in range(num_plots):
        axes[i].plot(x, y[i], colors[i], linewidth=0.75)
        axes[i].xaxis.set_tick_params(which='major', size=2, width=0.25, direction='in', right='on')
        axes[i].yaxis.set_tick_params(which='major', size=2, width=0.25, direction='in', right='on')
        axes[i].grid(True, linestyle=':')
        axes[i].set_yscale("log")
        axes[i].set_ylim(bottom=10E-3)
    
    fig.legend(labels=subtitles, loc="lower left", ncol=2, bbox_to_anchor=legend_loc, borderaxespad=0.0)
    
    fig.tight_layout(pad=fig_padding)
    plt.tick_params(axis='y', which='minor')
    plt.savefig(savename+'.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.show()
    
#%% 
"Question 1"
# Parameters
dt = 0.01
Omega_0 = 2*np.pi
t=np.arange(0.0, 5.0, dt) 

# Experiments
y_euler = OBE(EulerForward, rwa, t, dt, continuous_wave, Omega_0, 0.0, 0.0)
y_rk4 = OBE(RK4, rwa, t, dt, continuous_wave, Omega_0, 0.0, 0.0)
y_exact = Analytical_OBE(t, dt, Omega_0)

# Plots
subtitles = ['Euler', 'RK4', 'Analytical']
xtitle = 'Time (s)'
ytitle = 'Population Density of Excited State (${n_e}$)'
savename = 'Q1a'
multi_plot(t, [y_euler[:,2], y_rk4[:,2], y_exact], subtitles, xtitle, ytitle, savename, xlabel_pos=(0.56, -0.04), ylabel_pos=(-0.05, 0.58), fig_size=(2.9, 2), legend_loc=(0.0, 1.00))

subtitles = ['Euler', 'RK4']
xtitle = 'Time (s)'
ytitle = 'Error of Population Density of Excited State (${n_e}$)'
savename = 'Q1b'
y_euler_error = (y_exact-y_euler[:,2])**2
y_rk4_error = (y_exact-y_rk4[:,2])**2
multi_plot(t, [y_euler_error, y_rk4_error], subtitles, xtitle, ytitle, savename, xlabel_pos=(0.56, -0.04), ylabel_pos=(-0.05, 0.52), fig_size=(2.9, 2), legend_loc=(0.25, 1.00))


#%%
"Question 2"
# Parameters
dt = 0.01
t=np.arange(0.0, 10.0, dt) 
Omega_0 = 2*np.sqrt(np.pi)

# Experiment
y = OBE(RK4, rwa, t, dt, gaussian, Omega_0, 0.0, 0.0)

# Plot
xtitle = 'Time (s)'
ytitle = 'Population Density\n of Excited State (${n_e}$)'
savename = 'Q2'
single_plot(t, y[:,2], xtitle, ytitle, savename, ylabel_pos=(-0.01, 0.50))

#%%
"Question 3"

"Question 3a"
# Parameters 
dt = 0.01
t=np.arange(0.0, 10.0, dt) 
Omega_0 = 2*np.sqrt(np.pi)
num_pts = 100
detuning= np.linspace(0, Omega_0, num_pts)
peak_detuning = np.zeros(num_pts)

# Experiments
for i in range(num_pts): 
    y = OBE(RK4, rwa, t, dt, gaussian, Omega_0, detuning[i], 0.0)
    peak_detuning[i] = np.max(y[:,2])

"Question 3b"
# Parameters 
dt = 0.01
t=np.arange(0.0, 10.0, dt) 
Omega_0 = 2*np.sqrt(np.pi)
num_pts = 100
dephasing= np.linspace(0, Omega_0, num_pts)
peak_dephasing = np.zeros(num_pts)

# Experiments
for i in range(num_pts): 
    y = OBE(RK4, rwa, t, dt, gaussian, Omega_0, 0.0, dephasing[i])
    peak_dephasing[i] = np.max(y[:,2])
    
"Plotting Question 3"
x = [detuning, dephasing]
y = [peak_detuning, peak_dephasing]
xtitles = ['Detuning ($\Delta_{0L}$)', 'Dephasing ($\gamma_d$)']
ytitle = 'Peak Population Density\n of Excited State (${n_e}$)'
savename = 'Q3'
multi_plot_2(x, y, xtitles, ytitle, savename, ylabel_pos=(-0.05, 0.6), fig_size=(3.0, 1.5))

#%%
"Question 4a part 1"
# Parameters
dt = 0.01
t=np.arange(0.0, 10.0, dt) 
Omega_0 = 2*np.sqrt(np.pi)
omega_Ls = np.array([2, 5, 10, 20])*Omega_0

# Experiments
y = [[],[],[]]
for omega_L in omega_Ls: 
    y_out = OBE(RK4, non_rwa, t, dt, gaussian, Omega_0, omega_L, 0.0, 0.0)
    for i in range(3): y[i].append(y_out[:,i]) 
    
y_out = OBE(RK4, rwa, t, dt, gaussian, Omega_0, 0.0, 0.0)
for i in range(3): y[i].append(y_out[:,i]) 

# Plots 
subtitles = [r'${\omega_L}$=2${\Omega_0}$', 
             r'${\omega_L}$=5${\Omega_0}$', 
             r'${\omega_L}$=10${\Omega_0}$', 
             r'${\omega_L}$=20${\Omega_0}$', 
             'RWA']

xtitles = 3*['Time (s)']
ytitles = ['Real Coherence (Re(u))', 'Imaginary Coherence (Im(u))', 'Population Density of Excited State (${n_e}$)']
savename = 'Q4a'

multi_plot_3(t, y, subtitles, xtitles, ytitles, savename, legend_loc=(0.12, 1.0), fig_size=(7, 3.5))

#%% 
"Question 4a part 2"
# Parameters
dt = 0.001
t=np.arange(0.0, 10.0, dt) 
Omega_0 = 2*np.sqrt(np.pi)
omega_L = 4*np.sqrt(np.pi) 
phi = np.pi/2

# Experiments
y=[[],[],[]]
y_no_phase = OBE(RK4, non_rwa, t, dt, gaussian, Omega_0, omega_L, 0.0, 0.0)
y_phase = OBE(RK4, non_rwa, t, dt, gaussian, Omega_0, omega_L, 0.0, phi)
for i in range(3): y[i].append(y_no_phase[:,i]) 
for i in range(3): y[i].append(y_phase[:,i]) 

# Plots
subtitles = ['No Phase', r'$\dfrac{\pi}{2}$ Phase']
xtitle = 3*['Time (s)']
ytitles = ['Real Coherence (Re(u))', 'Imaginary Coherence (Im(u))', 'Population Density of Excited State (${n_e}$)']
savename = 'Q4a2'
multi_plot_3(t, y, subtitles, xtitles, ytitles, savename, legend_loc=(0.35, 1.0), fig_size=(7, 2.5), y_title_space=(0.01,0.35))
#%%  
"Question 4b"
# Parameters
dt = 0.01
t=np.arange(0.0, 10.0, dt) 
Omega_0s = np.array([2, 10, 20])*np.sqrt(np.pi)
omega_L = 4*np.sqrt(np.pi)

# Experiments
y=[[],[],[]]
for Omega in Omega_0s:
    y_out = OBE(RK4, non_rwa, t, dt, gaussian, Omega, omega_L, 0.0, 0.0)
    for i in range(3): y[i].append(y_out[:,i]) 

# Plots
subtitles = ['Area= 2$\pi$', 'Area= 10$\pi$', 'Area= 20$\pi$']
xtitle = 3*['Time (s)']
ytitles = ['Real Coherence (Re(u))', 'Imaginary Coherence (Im(u))', 'Population Density of Excited State (${n_e}$)']
savename = 'Q4b'
multi_plot_3(t, y, subtitles, xtitles, ytitles, savename, legend_loc=(0.265, 1.0), fig_size=(7, 3.5))

#%% 
"Question 4c"
# Parameters
dt = 0.001
t=np.arange(0.0, 50.0, dt) 
Omega_0s = np.array([2, 10, 20])*np.sqrt(np.pi)
omega_L = 4*np.sqrt(np.pi)
Dephasing = 0.4
max_w = 6

yf = []
w = 2*np.pi*np.fft.rfftfreq(len(t), dt)
w = w/(omega_L)
wf= w[w<max_w]

# Experiments
for Omega_0 in Omega_0s:
    y = OBE(RK4, non_rwa, t, dt, gaussian, Omega_0, omega_L, Dephasing, 0.0)
    y = np.abs(np.fft.rfft(y[:,0], norm='ortho'))
    y = y[w<max_w]
    yf.append(y)

# Plots
subtitles = ['Area= 2$\pi$', 'Area= 10$\pi$', 'Area= 20$\pi$']
xtitle = 'Frequency ($\omega/{\omega_L}$)'
ytitle = 'Power Spectrum of Polarization, |P($\omega$)|'
savename = 'Q4c'
multi_semilogy(wf,yf,subtitles, xtitle, ytitle, savename, ylabel_pos=(-0.03, 0.53), legend_loc=(0.15, 1.0), fig_size=(3, 4))
#%% 
"Question 4d"
# Parameters
dt = 0.01
t=np.arange(0.0, 10.0, dt) 
y0 = [0.0, 0.0, 0.0]
Omega_0 = 2*np.sqrt(np.pi)
omega_L = 4*np.sqrt(np.pi)
Dephasing = 0.4

# Experiments
num_trials = 20
average = 0
for i in range(num_trials): 
    start = timeit.default_timer()
    y=odeint(non_rwa, y0, t, args=(gaussian, Omega_0, omega_L, Dephasing, 0))
    stop = timeit.default_timer()
    dt_scipy = stop-start
    
    start = timeit.default_timer()
    y=OBE(RK4, non_rwa, t, dt, gaussian, Omega_0, omega_L, Dephasing, 0.0)
    stop = timeit.default_timer()
    dt_custom = stop-start

    average += (dt_custom/dt_scipy)/num_trials

print("Odeint is %.2f times faster than RK4." % average)

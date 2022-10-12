# -*- coding: utf-8 -*-
"""
Late edited: March 2022

@author: Andrew Hayman
"""

"""
FDTD simulation for 1D. The program allows the user to define a vacuum slab
with a certain spacing between space discretized points. Simple absorbing 
boundary conditions and total-field scattered-field are both implemented. 
The user can add in slabs of different types and parameters anywhere within 
the vacuum slab. Here, we implement dielectric slabs, Drude dispersion model 
slabs, and Lorentz dispersion model slabs. Finally, the user can add any time 
dependent pulse and here we show an example with a sinusoidal pulse with a 
Gaussian envelope. 

A variety of plot settings can be applied. In particular, the field over time, 
frequency, reflection/tranmission, and snapshots can all be shown. An 
animation feature is included to show the field evolution dynamically. 

Note that everything has been changed to physical parameters to make this
more relevant to physical systems. 

Instructions: 
1. Scroll down to the question section and set any desired plot settings
2. If you turn on animation, be sure to set %matplotlib auto in the console
3. Run each question one by one. Feel free to change parameters to experiment
with other scenarios. 

"""
#%% Imports
import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.constants as constants
import matplotlib as mpl

#%% Matplotlib settings
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5

#%% Constants
"Constants used throughout the code."
femto = constants.femto
tera = constants.tera
micro = constants.micro
nano = constants.nano
peta = constants.peta

#%% Custom Pulses 
class Sine_Gaussian():
    """Gaussian envelope around a sinusoidal signal."""
    def __init__(self, dt, duration, frequency):
        """Accepts the time between time discretized points, 
        the pulse duration in fs and the pulse frequency in THz. 
        All parts of pulse are scaled by dt such that the pulse can accept 
        indices instead of specific time points."""
        self.spread = duration / dt 
        self.t0 = self.spread * 6
        freq_in = 2 * math.pi * frequency
        self.w_scale = freq_in * dt
        self.lam = 2*math.pi*constants.c/freq_in
        
    def __call__(self, t):
        """Returns the pulse for given time index."""
        return -np.exp(-0.5 * (t - self.t0) ** 2 / self.spread ** 2) * (np.cos(t * self.w_scale))
    
    
class Drude(): 
    """Drude dispersion model."""
    def __init__(self, alpha, w_p): 
        self.alpha = alpha
        self.w_p = w_p
        self.name = "drude"
        self.Sn1= 0
        self.Sn2 = 0
    
    def get_epsilon(self, w): 
        num = self.w_p**2
        denom = w**2+1j*w*self.alpha
        epsilon = 1 - num/denom
        return epsilon
        
    def get_S(self, En1, delta_t): 
        """Returns the E correction based on the Z-transform.
        S is stored for the previous two iterations."""
        exp = np.exp(-self.alpha*delta_t)
        term1 = (1+exp)*self.Sn1
        term2 = -exp*self.Sn2
        term3 = (delta_t*self.w_p**2/self.alpha)*(1-exp)*En1  
        S = term1 + term2 + term3
        
        if( not isinstance(self.Sn1,int)): 
            self.Sn2 = self.Sn1
        self.Sn1 = S
        return S
    
    def __call__(self, Dx, Ex_prev, a, b, dt): 
        """Find next E field."""
        S = self.get_S(Ex_prev[a:b], dt)
        Dx[a:b] -= S
        return Dx
        
class Lorentz(): 
    """Lorentz dispersion model"""
    def __init__(self, alpha, w_0, f_0): 
        self.alpha = alpha
        self.w_0 = w_0
        self.f_0 = f_0
        self.B = np.sqrt(w_0**2-f_0**2)
        self.name = "lorentz"
        self.Sn1= 0
        self.Sn2 = 0    
    
    def get_epsilon(self, w): 
        num = self.f_0*self.w_0**2
        denom = self.w_0**2-w**2-1j*2*w*self.alpha
        epsilon = 1 + num/denom
        return epsilon
    
    def get_S(self, En1, delta_t): 
        """Returns the E correction based on the Z-transform.
        S is stored for the previous two iterations."""
        exp1 = np.exp(-self.alpha*delta_t)
        exp2 = np.exp(-2*self.alpha*delta_t)
        term1 = 2*exp1*np.cos(self.B*delta_t)*self.Sn1
        term2 = -exp2*self.Sn2
        term3_coeff = delta_t*self.f_0*(self.B+self.alpha**2/self.B)
        term3 = term3_coeff*exp1*np.sin(self.B*delta_t)*En1
        S = term1 + term2 + term3
        
        if( not isinstance(self.Sn1,int)): 
            self.Sn2 = self.Sn1
        self.Sn1 = S
    
        return S
    
    def __call__(self, Dx, Ex_prev, a, b, dt): 
        """Find next E field."""
        S = self.get_S(Ex_prev[a:b], dt)
        Dx[a:b] -= S
        return Dx

class Dielectric(): 
    def __init__(self, epsilon): 
        self.epsilon = epsilon 
        self.name="dielectric"
    
    def get_epsilon(self, w=0): 
        return self.epsilon        
    
    def __call__(self, Dx, Ex_prev, a, b, dt): 
        """Find next E field."""
        Dx[a:b] /= self.epsilon
        return Dx

#%% Main Experiment
class FDTD():
    """Our FDTD experiment assumes the B=0.5. The user supplies the dx spacing 
    and the grid sizes. """
    def __init__(self, length, dx):
        """Creates vacuum slab of a certain length space discretized by 
        some amount dx. The time spacing dt is found based on the Beta
        stability condition assuming B=0.5."""
        self.dx = dx
        self.length = length
        
        self.x_pts = int(self.length/self.dx + 1)
        self.dt = self.dx/(2.*constants.c)
        
        # Initially set no material
        self.material = None
        
    def insert_slab(self, length, position, material): 
        """Insert a slab of some material type and thickness at a desired
        position."""
        self.material = material
        self.slab_length= length
        self.slab_position = position
        
        self.slab_n_pts = int(self.x_pts*length/self.length)
        self.slab_start_idx = int(self.x_pts*position/self.length)
        self.slab_end_idx = self.slab_start_idx+self.slab_n_pts

    def insert_pulse(self, position, pulse, duration, frequency):
        """Insert a pulse of some type at the desired position. A pulse
        duration and frequency are required."""
        self.pulse_position = position
        self.pulse = pulse(self.dt, duration, frequency) 
        self.pulse_idx = int(self.x_pts*position/self.length)
        
    def analytical_R(self, w): 
        """Find analytical reflection based on frequency and the 
        material's dispersion relation."""
        n = np.sqrt(self.material.get_epsilon(w))
        r1 = (1-n)/(1+n)
        r2 = (n-1)/(n+1)
        k_0 = w/constants.c
        
        num = r1+r2*np.exp(2j*k_0*self.slab_length*n)
        denom = 1+r1*r2*np.exp(2j*k_0*self.slab_length*n)
        r_w = num/denom
        R = np.abs(r_w)**2
        return R
        
    def analytical_T(self, w): 
        """Find analytical transmission based on frequency and the 
        material's dispersion relation."""
        n = np.sqrt(self.material.get_epsilon(w))
        r1 = (1-n)/(1+n)
        r2 = (n-1)/(n+1)
        k_0 = w/constants.c
        
        num = (1+r1)*(1+r2)*np.exp(1j*k_0*self.slab_length*n)
        denom = 1+r1*r2*np.exp(2j*k_0*self.slab_length*n)
        t_w = num/denom
        T = np.abs(t_w)**2
        return T
        
    def set_plot_settings(self, animate=False, checkpoints=[], plot_name="", save=False, track_points=[], time_plot=True):
        """Set plot settings."""
        self.animate = animate
        self.checkpoints = [int(checkpoint/femto) for checkpoint in checkpoints]
        self.plot_name = plot_name
        self.save = save
        self.track_points = track_points
        self.time_plot = time_plot
        
    def check_lambda(self):
        """Checks wavelength against space intervals."""
        if self.material and self.material.name=="dielectric": 
            n = np.sqrt(self.material.get_epsilon())
        else: 
            n = 1
        ppw = int(self.pulse.lam/(n*self.dx))
        print('points per wavelength',ppw, 'should be > 15')
        
    def solve(self, time, w): 
        """Solve for fields over a given period of time and compute running
        FT of given frequencies."""
        
        # Check lambda
        self.check_lambda()
        
        # Time values
        t_steps = int(time/self.dt)
        t_vals = np.arange(0,time, self.dt)
        
        # Arrays of space discretized fields
        Dx = np.zeros((self.x_pts), float)  
        Ex = np.zeros((self.x_pts), float)  
        Hy = np.zeros((self.x_pts), float)
        
        # Arrays to hold running FFT
        E_t_ft = np.zeros(len(w), dtype=complex)
        E_r_ft = np.zeros(len(w), dtype=complex)
        E_in_ft = np.zeros(len(w), dtype=complex)
        
        # Arrays to hold any tracking points
        E_track = [np.zeros((t_steps+1), float) for i in range(len(self.track_points))]
        if self.time_plot: 
            E_time = [np.zeros((t_steps+1), float) for i in range(3)]
        
        # Arrays to hold BC
        Ex_2 = np.zeros(2)
        Ex_end = np.zeros(2)
        
        # Setup plot 
        if self.animate: 
            self.setup_plot(Ex)
        
        # Iterate over time
        for i in range(0, t_steps + 1):
            t = i-1 
            
            # Pulses
            pulseH = self.pulse(t+1/2)
            pulseE = self.pulse(t)
            
            # Update dielectric coefficients
            Dx[1:] += 0.5*(Hy[:-1]-Hy[1:])
            Dx[self.pulse_idx] -= 0.5*pulseH
            
            # Get E field based on material type
            if self.material: 
                Ex = self.material(Dx.copy(), Ex, self.slab_start_idx, self.slab_end_idx, self.dt)
            else: 
                Ex = Dx.copy()
                
            # Apply BC
            Ex[0] = Ex_2[0]
            Ex_2[0] = Ex_2[1]
            Ex_2[1] = Ex[1]
            
            Ex[-1] = Ex_end[0]
            Ex_end[0] = Ex_end[1]
            Ex_end[1] = Ex[-2]

            # Update H field vectorized
            Hy[:-1] += 0.5*(Ex[:-1]-Ex[1:])
            Hy[self.pulse_idx-1] -= 0.5*pulseE

            # Compute running FT
            if self.material: 
                E_t_ft += Ex[-1]*(np.cos(self.dt*w*t)-1j*np.sin(self.dt*w*t))
                E_r_ft += Ex[0]*(np.cos(self.dt*w*t)-1j*np.sin(self.dt*w*t))
                E_in_ft += self.pulse(t)*(np.cos(self.dt*w*t)-1j*np.sin(self.dt*w*t))
            
            # Track points
            for j in range(len(E_track)):
                E_track[j][i] = Ex[int(self.track_points[j]/self.dx)]
            if self.time_plot: 
                E_time[0][i] = self.pulse(t)
                E_time[1][i] = Ex[0]
                E_time[2][i] = Ex[-1]
            
            # Update plot
            if self.animate and i % 10 == 0: 
                self.update_plot(Ex, t_vals[i]) 
                plt.pause(10/200)     
            else: 
                self.plot_snapshot(Ex, t_vals[i])               

        # Compute final FT
        if self.material: 
            E_t_ft = np.abs(E_t_ft)/np.sqrt(t_steps + 1)
            E_r_ft = np.abs(E_r_ft)/np.sqrt(t_steps + 1)
            E_in_ft = np.abs(E_in_ft)/np.sqrt(t_steps + 1)
            self.plot_frequency_signals(w, E_r_ft, E_t_ft, E_in_ft)
            self.plot_power_spectrum(w, E_r_ft, E_t_ft, E_in_ft)
          
        # Plot time signals
        self.plot_time_signals(t_vals, E_track)
        if self.time_plot: 
            self.plot_E_time(t_vals, E_time)
    
    def plot_power_spectrum(self, w_ft, E_r_ft, E_t_ft, E_in_ft): 
        """Plot the reflection, transmission and sum over desired frequency 
        range along with the analytical solutions.""" 
        R = np.abs(E_r_ft)**2/np.abs(E_in_ft)**2
        T = np.abs(E_t_ft)**2/np.abs(E_in_ft)**2
        Sum = R+T
               
        R_an = self.analytical_R(w_ft)
        T_an = self.analytical_T(w_ft)
        Sum_an = R_an+T_an

        fig, axes = plt.subplots(1, figsize=(2.5, 2.5))
        axes.plot(w_ft/(2*np.pi*tera), R_an, color='r', ls='--', lw=0.8)
        axes.plot(w_ft/(2*np.pi*tera), T_an, color='b', ls='--', lw=0.8)
        axes.plot(w_ft/(2*np.pi*tera), Sum_an, color='g', ls='--', lw=0.8)
        axes.plot(w_ft/(2*np.pi*tera), R, color='r', lw=0.8)
        axes.plot(w_ft/(2*np.pi*tera), T, color='b', lw=0.8)
        axes.plot(w_ft/(2*np.pi*tera), Sum, color='g', lw=0.8)
        axes.grid(True, linestyle=':')
        axes.set_xlabel(r'Frequency, $\omega/2\pi$ (THz)')
        axes.set_ylabel("Field Strength")
        fig.legend(labels=[r'$R_{an}$', r'$T_{an}$', r'$Sum_{an}$', r'$R$', r'$T$', r'$Sum$'], 
                   ncol=2, 
                   bbox_to_anchor=(0.95,1.23), 
                   borderaxespad=0.)
        fig.tight_layout()
        if self.save:
            plt.savefig(self.plot_name + "_power" + ".pdf", format='pdf', dpi=1200,bbox_inches = 'tight')
        plt.show()
        
    def plot_frequency_signals(self, w_ft, E_r_ft, E_t_ft, E_in_ft):   
        """Plot the frequency signals over the desired range of frequencies."""
        fig, axes = plt.subplots(1, figsize=(2.5, 2.5))
        axes.plot(w_ft/(2*np.pi*tera), E_r_ft)
        axes.plot(w_ft/(2*np.pi*tera), E_t_ft)
        axes.plot(w_ft/(2*np.pi*tera), E_in_ft)
        axes.grid(True, linestyle=':')
        axes.set_xlabel(r'Frequency, $\omega/2\pi$ (THz)')
        axes.set_ylabel("Frequency Strength")
        fig.legend(labels=[r'$E_r(\omega)$', r'$E_t(\omega)$', r'$E_{in}(\omega)$'], 
                   ncol=2, 
                   bbox_to_anchor=(0.95,1.15), 
                   borderaxespad=0.)
        fig.tight_layout()
        if self.save: 
            plt.savefig(self.plot_name + "_frequency" + ".pdf", format='pdf', dpi=1200,bbox_inches = 'tight')
        plt.show()
        
    def plot_time_signals(self, t, Es): 
        """Plot the field at some desired point over time."""
        if Es: 
            fig, axes = plt.subplots(1, figsize=(3.2, 2.8))
            t = t/femto
            for E in Es: 
                axes.plot(t,E)
            axes.grid(True, linestyle=':')
            axes.set_xlabel("Time (fs)")
            axes.set_ylabel("Field")
            fig.tight_layout()
            if self.save:
                plt.savefig(self.plot_name + "_time_track" + ".pdf", format='pdf', dpi=1200,bbox_inches = 'tight')
            plt.show() 
            
    def plot_E_time(self, t, Es): 
        """Plot the field over time."""
        fig, axes = plt.subplots(1, figsize=(2.5, 2.5))
        t = t/femto
        for E in Es: 
            print(np.max(E))
            axes.plot(t,E)
        axes.grid(True, linestyle=':')
        axes.set_xlabel("Time (fs)")
        axes.set_ylabel("Field")
        fig.legend(labels=[r'$E_{in}(t)$', r'$E_r(t)$', r'$E_t(t)$'], 
                   ncol=2, 
                   bbox_to_anchor=(0.95,1.15), 
                   borderaxespad=0.)
        fig.tight_layout()
        if self.save:
            plt.savefig(self.plot_name + "_time" + ".pdf", format='pdf', dpi=1200,bbox_inches = 'tight')
        plt.show() 
            
    def plot_snapshot(self, Ex, t): 
        """Plot a single snapshot at the desired time. Due to time steps, the 
        nearest time step to the desired snapshot is selected."""
        if(int(t/femto) in self.checkpoints and self.save): 
            self.setup_plot(Ex)
            self.checkpoints = self.checkpoints[1:]
            plt.savefig(self.plot_name + "_" + str(int(t/femto)) + ".pdf", 
                        format='pdf', 
                        dpi=1200,
                        bbox_inches = 'tight')
            plt.show()
        
    def update_plot(self, Ex, t): 
        """Update data in plot for animation if animation is activated."""
        self.im.set_ydata(Ex[1:self.x_pts-1])
        self.ax.set_title("Time %.2f fs"%(t/femto))
        if(int(t/femto) in self.checkpoints and self.save): 
            self.checkpoints = self.checkpoints[1:]
            plt.savefig(self.plot_name + "_" + str(int(t/femto)) + ".pdf", 
                        format='pdf', 
                        dpi=1200,
                        bbox_inches = 'tight')
        plt.show()
        
    def setup_plot(self, Ex): 
        """Setup plot for animation or snapshots."""
        self.fig = plt.figure(figsize=(2.5,2.5))
        self.ax = self.fig.add_axes([.18, .18, .7, .7])
        self.im = self.ax.plot(np.arange(self.dx,self.length,self.dx)/micro,Ex[1:self.x_pts-1],linewidth=1)[0]
        plt.ylim((-1.1, 1.1))
        
        if self.material: 
            plt.axvline(x=(self.slab_position)/micro,color='r',linewidth=1)
            plt.axvline(x=(self.slab_position+self.slab_length)/micro,color='r',linewidth=1)
            
        plt.axvline(x=self.pulse_position/micro,color='b',linewidth=1)
            
        plt.axvline(x=0, color='black', ls='--',linewidth=1)
        plt.axvline(x=self.length/micro, color='black', ls='--',linewidth=1)
            
        plt.grid('on')
        self.ax.grid(True, linestyle=':')
        self.ax.set_xlabel(r'Position, x ($\mu m$)')
        self.ax.set_ylabel('$E_x$', labelpad=-5)
   
#%% Plot settings for all questions 
checkpoints=femto*np.arange(0,45,5) # =[]
animate = False # True
save = True # False

#%% Q1a), Q1b)
# No Slab
# Pulse: 2fs at 200THz at 2um
fdtd = FDTD(length = 8*micro, dx = 20*nano)
 
fdtd.insert_pulse(position = 2*micro, 
                  pulse = Sine_Gaussian, 
                  duration = 2*femto, 
                  frequency = 200*tera)

fdtd.set_plot_settings(animate=animate, 
                       checkpoints=checkpoints, 
                       plot_name="plots/No_Slab", 
                       save=save, 
                       track_points=[1*micro])

fdtd.solve(time = 80*femto, w=2*np.pi*tera*np.arange(100,300,1))

#%% Q1c), Q1d) 
#Dielectric Slab
# 8um slab, 20nm interval
# Slab: 1um slab from 3um-4um with dielectric of epsilon=9
# Pulse: 2fs at 200THz at 2um
#checkpoints=femto*np.arange(0,45,5)
fdtd = FDTD(length = 8*micro, dx = 20*nano) 

fdtd.insert_slab(length=1*micro, 
                 position=3*micro, 
                 material = Dielectric(epsilon=9))

fdtd.insert_pulse(position = 2*micro, 
                  pulse = Sine_Gaussian, 
                  duration = 2*femto, 
                  frequency = 200*tera)

fdtd.set_plot_settings(animate=animate, 
                       checkpoints=checkpoints, 
                       plot_name="plots/Dielectric", 
                       save=save)

fdtd.solve(time = 150*femto, w=2*np.pi*tera*np.arange(100,300,1))

#%% Q2 b) 
#Drude Model Slab 
# 8um slab, 20nm interval
# Slab: 200nm slab from 3um to 3.2um with Drude model 
# of alpha=140/2pi THz, plasma frequency 1.26/2pi PHz
# Pulse: 1fs at 200THz at 2um
fdtd = FDTD(length = 8*micro, dx = 20*nano) 

fdtd.insert_slab(length=200*nano, 
                 position=3*micro, 
                 material = Drude(alpha=140*tera, w_p=1.26*peta))

fdtd.insert_pulse(position = 2*micro, 
                  pulse = Sine_Gaussian, 
                  duration = 1*femto, 
                  frequency = 200*tera)

fdtd.set_plot_settings(animate=animate, 
                       checkpoints=checkpoints, 
                       plot_name="plots/Drude_200", 
                       save=save)

fdtd.solve(time = 150*femto, w=2*np.pi*tera*np.arange(0,600,1))

#%% Q2 b) ii)
#Drude Model Slab 
# 8um slab, 20nm interval
# Slab: 800nm slab from 3um to 3.8um with Drude model 
# of alpha=140/2pi THz, plasma frequency 1.26/2pi PHz
# Pulse: 1fs at 200THz at 2um
fdtd = FDTD(length = 8*micro, dx = 20*nano) 

fdtd.insert_slab(length=800*nano, 
                 position=3*micro, 
                 material = Drude(alpha=140*tera, w_p=1.26*peta))

fdtd.insert_pulse(position = 2*micro, 
                  pulse = Sine_Gaussian, 
                  duration = 1*femto, 
                  frequency = 200*tera)

fdtd.set_plot_settings(animate=animate, 
                       checkpoints=checkpoints, 
                       plot_name="plots/Drude_800", 
                       save=save)

fdtd.solve(time = 150*femto, w=2*np.pi*tera*np.arange(0,600,1))

#%% Q2 c) i)
#Lorentz Model Slab 
# 8um slab, 20nm interval
# Slab: 200nm slab from 3um to 3.2um with Lorentz model 
# of alpha=2THz, resonant frequency=200THz, f_0=0.05
# Pulse: 1fs at 200THz at 2um
fdtd = FDTD(length = 8*micro, dx = 20*nano) 

fdtd.insert_slab(length=200*nano, 
                 position=3*micro, 
                 material = Lorentz(alpha=4*np.pi*tera, w_0=2*np.pi*200*tera, f_0=0.05))

fdtd.insert_pulse(position = 2*micro, 
                  pulse = Sine_Gaussian, 
                  duration = 1*femto, 
                  frequency = 200*tera)

fdtd.set_plot_settings(animate=animate, 
                       checkpoints=checkpoints, 
                       plot_name="plots/Lorentz_200", 
                       save=save)

fdtd.solve(time = 150*femto, w=2*np.pi*tera*np.arange(150,250,1))
        
#%% Q2 c) ii)
#Lorentz Model Slab 
# 8um slab, 20nm interval
# Slab: 800nm slab from 3um to 3.8um with Lorentz model 
# of alpha=2THz, resonant frequency=200THz, f_0=0.05
# Pulse: 1fs at 200THz at 2um
fdtd = FDTD(length = 8*micro, dx = 20*nano) 

fdtd.insert_slab(length=800*nano, 
                 position=3*micro, 
                 material = Lorentz(alpha=4*np.pi*tera, w_0=2*np.pi*200*tera, f_0=0.05))

fdtd.insert_pulse(position = 2*micro, 
                  pulse = Sine_Gaussian, 
                  duration = 1*femto, 
                  frequency = 200*tera)

fdtd.set_plot_settings(animate=animate, 
                       checkpoints=checkpoints, 
                       plot_name="plots/Lorentz_800", 
                       save=save)

fdtd.solve(time = 150*femto, w=2*np.pi*tera*np.arange(150,250,1))


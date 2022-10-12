# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 20:56:30 2022

@author: Andrew Hayman

This program examines the 1D and 2D Ising model using the Metropolis 
algorithm and Monte Carlo sampling.

Instructions:
Run each question sequentially. Written answers are included as comments in
code section for Question 1a) and 1b). It is easy to adjust any parameters
to try different configurations if desired. 

Please run any numba code twice to see the speedup from caching. Numba is 
always used in the initialize and update functions. It is only used for the 
main loop in the kT sweep for the 2d case to make plotting simpler. On my
computer, for 100 kT values, I get 7.5 seconds for a 20x20 lattice. 
"""

#%% Imports
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib as mpl
import numba 
import timeit

#%% Matplotlib settings
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5

#%% Analytical Expectation Values 
def get_E_exp_an(kT): 
    """Get the analytical expectation value of E for N spins and kT 
    temperature. Returned values are normalized in N."""
    B = 1/kT
    return -np.tanh(B)

def get_S_exp_an(kT): 
    """Get the analytical expectation value of S for N spins and kT
    temperature. Returned values are normalized in Nk."""
    B = 1/kT
    return np.log(2*np.cosh(B))-B*np.tanh(B)

#%% 1D Ising Model
@numba.jit(nopython=True, fastmath=True, cache=True)
def initialize_1d(N, p): 
    """Initialize spin, energy and magnetization for N spins."""
    spin= np.ones(N)
    E = 0. 
    M = 0. 
    for i in range(1,N): 
        if np.random.rand(1) < p: 
            spin[i]=-1
        E = E - spin[i-1]*spin[i]
        M = M + spin[i]
        
    # Periodic BC
    E = E - spin[N-1]*spin[0]
    M = M + spin[0]
    return spin, E, M

@numba.jit(nopython=True, fastmath=True, cache=True)
def update_1d(N, spin, kT, E, M): 
    """Update spin using Metropolis anglorithm."""
    num = np.random.randint(0,N)
    dE = 2*spin[num]*(spin[num-1] + spin[(num+1)%N])
    
    if dE < 0.0 or np.random.rand(1) < np.exp(-dE/kT): 
        E += dE
        M -= 2*spin[num]
        spin[num] = -spin[num]

    return spin, E, M

def run_1d_simulation(N, iterations, kT, p, time=False, plot=False): 
    """Run the Metropolis algorithm for a given kT for N spins over
    N*iterations steps. Optional settings include printing the time 
    to completion and plotting the spin and energy."""
    if time: 
        start = timeit.default_timer() 
        
    n_steps = iterations*N
    
    spin = np.zeros((N, n_steps))
    E = np.zeros(n_steps)
    M = np.zeros(n_steps)

    spin[:,0], E[0], M[0] = initialize_1d(N, p)
    for i in range(1,n_steps): 
        spin[:,i], E[i], M[i] = update_1d(N, spin[:,i-1], kT, E[i-1], M[i-1])
    
    if time: 
        end = timeit.default_timer()
        print("Time: %.4f seconds"%(end-start))
        
    if plot: 
        E_exp_an = get_E_exp_an(kT)
        plot_spin_E_1d(n_steps, N, spin, E, E_exp_an, kT, p)
    
    return E, M

def run_1d_kT_sweep(N, iterations, kT, p, time=False, plot=False):  
    """Sweep the 1d simulation over a series of kT values. Optional settings
    include printing the time to completion and plotting the numerical 
    and analytical values of energy, magnetization and entropy over the given 
    kT values."""
    if time: 
        start = timeit.default_timer() 
        
    n_steps = iterations*N
    n_half = int(n_steps/2)
    n_kT = len(kT)
    
    E_exp_an = np.zeros(n_kT)
    E_exp_num = np.zeros(n_kT)
    M_exp_num = np.zeros(n_kT)
    S_exp_an = np.zeros(n_kT)
    S_exp_num = np.zeros(n_kT)
    
    for i in range(n_kT):
        E, M = run_1d_simulation(N, iterations, kT[i], p, time=False, plot=False)
           
        E_exp_num[i] = np.sum(E[n_half:])/n_half
        M_exp_num[i] = np.sum(M[n_half:])/n_half     
        if(i>0): 
            dE = E_exp_num[i]-E_exp_num[i-1]
            S_exp_num[i] = S_exp_num[i-1] + dE/kT[i]
        
        E_exp_an[i] = get_E_exp_an(kT[i])
        S_exp_an[i] = get_S_exp_an(kT[i])

        
    if time: 
        end = timeit.default_timer()
        print("Time: %.4f seconds"%(end-start))
        
    if plot:  
        plot_E_T_1d(kT, E_exp_num, E_exp_an, N)
        plot_M_T_1d(kT, M_exp_num, N)
        plot_S_T_1d(kT, S_exp_num, S_exp_an, N)

def plot_spin_E_1d(n_steps, N, spin, E, E_an, kT, p):
    """Plot the spin evolution and enery evolution over the
    number of iterations."""
    fig, axes = plt.subplots(2, figsize=(3.5, 4))
    X, Y = np.meshgrid(np.arange(n_steps)/N, np.arange(N))
    axes[0].set_title("kT= " + str(kT) + " and p= " + str(p))
    axes[0].pcolormesh(X, Y, spin, cmap=plt.cm.RdBu);
    axes[0].set_ylabel("N Spins")
    
    x = np.arange(len(E))
    axes[1].set_xlabel("Iteration/N")
    axes[1].set_ylabel(r'Energy/N$\epsilon$')
    axes[1].plot(x/N, E/N)
    axes[1].grid(True, linestyle=':')
    axes[1].hlines(E_an, 0, len(E)/N, color='red')
    axes[1].legend(["E", "<E> an"])
    fig.tight_layout()
    plt.show()
  
def plot_E_T_1d(kT, E_exp_num, E_exp_an, N):
    """Plot the numerical and analytical expectation value of E."""
    fig, axes = plt.subplots(1, figsize=(3.5, 2.5))
    axes.set_xlabel(r'kT/$\epsilon$')
    axes.set_ylabel(r'<E>/N$\epsilon$')
    axes.plot(kT, E_exp_num/N, 'o', ms=3)
    axes.plot(kT, E_exp_an)
    axes.grid(True, linestyle=':')
    axes.legend(["<E> num", "<E> an"])
    plt.show()
    
def plot_M_T_1d(kT, M_exp_num, N):
    """Plot the numerical and analytical expectation value of M."""
    fig, axes = plt.subplots(1, figsize=(3.5, 2.5))
    axes.set_xlabel(r'kT/$\epsilon$')
    axes.set_ylabel(r'<M>/N$\mu$')
    axes.plot(kT, M_exp_num/N, 'o', ms=3)
    axes.hlines(0, np.min(kT), np.max(kT), color='red') # Analytical M=0
    axes.legend(["<M> num", "<M> an"])
    axes.grid(True, linestyle=':')
    plt.show()
    
def plot_S_T_1d(kT, S_exp_num, S_exp_an, N):
    """Plot the numerical and analytical expectation value of S."""
    fig, axes = plt.subplots(1, figsize=(3.5, 2.5))
    axes.set_xlabel(r'kT/$\epsilon$')
    axes.set_ylabel(r'<S>/Nk')
    axes.plot(kT, S_exp_num/N, 'o', ms=3)
    axes.plot(kT, S_exp_an)
    axes.grid(True, linestyle=':')
    axes.legend(["<S> num", "<S> an"])
    plt.show()
    
#%% 2D Ising Model
@numba.jit(nopython=True, fastmath=True, cache=True)
def initialize_2d(N, p): 
    """Initialize spin, energy and magnetization for NxN spin lattice."""
    spin= np.ones((N,N))
    E = 0. 
    M = 0. 
    for i in range(1,N): 
        for j in range(1,N): 
            if np.random.rand(1) < p: 
                spin[i][j]=-1
            E = E - spin[i-1][j]*spin[i][j] - spin[i][j-1]*spin[i][j]
            M = M + spin[i][j]
        
    # Periodic BC
    for i in range(1,N): 
        E = E - spin[i][N-1]*spin[i][0]
        M = M + spin[i][0]
        
    for i in range(0,N): 
        E = E - spin[N-1][i]*spin[0][i]
        M = M + spin[0][i]
        
    return spin, E, M

@numba.jit(nopython=True, fastmath=True, cache=True)
def update_2d(N, spin, kT, E, M): 
    """Update spin using Metropolis anglorithm."""
    num1 = np.random.randint(0,N)
    num2 = np.random.randint(0,N)
    dE = 2*spin[num1][num2]*(spin[num1-1][num2] + 
                             spin[(num1+1)%N][num2] + 
                             spin[num1][num2-1] + 
                             spin[num1][(num2+1)%N])
        
    if dE < 0.0 or np.random.rand(1) < np.exp(-dE/kT): 
        E += dE
        M -= 2*spin[num1][num2]
        spin[num1][num2] = -spin[num1][num2]

    return spin, E, M
    
def run_2d_simulation(N, iterations, kT, p, time=False, plot=False, frames=[]): 
    """Run the Metropolis algorithm for a given kT for N spins over
    N*iterations steps. Optional settings include printing the time 
    to completion, plotting spin snapshots and plotting the energy evolution.
    """
    
    if time: 
        start = timeit.default_timer() 
        
    n_steps = N*N*iterations
    
    
    if plot: 
        n_half = int(n_steps/2)
        E = np.zeros(n_steps)
        M = np.zeros(n_steps)
        
        spin, E[0], M[0] = initialize_2d(N, p)
        for i in range(1,n_steps): 
            spin, E[i], M[i] = update_2d(N, spin, kT, E[i-1], M[i-1])
            
            if i in frames: 
                plot_spin_2d(N, N, spin, i)
        
        E_avg = np.sum(E[n_half:])/n_half
        M_avg = np.sum(M[n_half:])/n_half
        
        plot_E_2d(E, N)
        
    else: 
        @numba.jit(nopython=True, fastmath=True, cache=True)
        def run_sim(): 
            E_avg = 0
            M_avg = 0
            
            spin, E, M = initialize_2d(N, p)
            for i in range(1,n_steps): 
                spin, E, M = update_2d(N, spin, kT, E, M)
                if(i>=n_steps/2): 
                    E_avg += E
                    M_avg += M
        
            E_avg/=(n_steps/2)
            M_avg/=(n_steps/2)
            return E_avg, M_avg
        E_avg, M_avg = run_sim()
        
        
    
    if time: 
        end = timeit.default_timer()
        print("Time: %.4f seconds"%(end-start))
    
    return E_avg, M_avg

def run_2d_kT_sweep(N, T_Tc, p, time=False, plot=False):    
    """Sweep the 2d simulation over a series of T/Tc values for an NxN lattice. 
    Optional settings include printing the time to completion and plotting the 
    expectation values of energy and magnetization over the given T/Tc values.
    """
    
    Tc = 2/np.log(1+np.sqrt(2))
    kT = T_Tc*Tc
    
    if time: 
        start = timeit.default_timer() 

    n_kT = len(kT)
    E_exp = np.zeros(n_kT)
    M_exp = np.zeros(n_kT)
    
    for i in range(len(kT)):
        E_exp[i], M_exp[i] = run_2d_simulation(N, 2*N*N, kT[i], p, time=False, plot=False)
            
    if time: 
        end = timeit.default_timer()
        print("Time: %.4f seconds"%(end-start))
        
    if plot:  
        plot_E_T_2d(T_Tc, E_exp, N)
        plot_M_T_2d(T_Tc, M_exp, N)
        
def plot_spin_2d(x_len, y_len, spin, t):
    """Plot spin configuration snapshot."""
    fig, axes = plt.subplots(1, figsize=(3.5, 2.5))
    X, Y = np.meshgrid(range(x_len), range(y_len))
    axes.pcolormesh(X, Y, spin, cmap=plt.cm.RdBu);
    axes.set_xlabel("Grid Cells (x)")
    axes.set_ylabel("Grid Cells (y)")
    axes.set_title("Frame Time: " + str(t))
    plt.show()
    
def plot_E_2d(E, N):
    """Plot energy evolution over time."""
    fig, axes = plt.subplots(1, figsize=(3.5, 2.5))
    x = np.arange(len(E))
    axes.set_xlabel(r'Iteration/$N^2$')
    axes.set_ylabel(r'Energy/$N^2\epsilon$')
    axes.plot(x/(N**2), E/(N**2))
    axes.grid(True, linestyle=':')
    plt.show()
    
def plot_E_T_2d(kT, E_exp_num, N):
    """Plot the numerical expectation value of E."""
    fig, axes = plt.subplots(1, figsize=(3.5, 2.5))
    axes.set_xlabel(r'T/$\epsilon T_c$')
    axes.set_ylabel(r'<E>/$N^2\epsilon$')
    axes.plot(kT, E_exp_num/(N*N), 'o', ms=3)
    axes.set_ylim(-2.1, 0.1)
    axes.grid(True, linestyle=':')
    plt.show()
    
def plot_M_T_2d(kT, M_exp_num, N):
    """Plot the numerical expectation value of M."""
    fig, axes = plt.subplots(1, figsize=(3.5, 2.5))
    axes.set_xlabel(r'T/$\epsilon T_c$')
    axes.set_ylabel(r'|<M>|/$N^2\mu$')
    axes.plot(kT, np.abs(M_exp_num)/(N*N), 'o', ms=3)
    axes.grid(True, linestyle=':')
    plt.show()
    
#%% Question 1a) kT=0.1, 0.5, 1. Discussion of time to equilibrium for
# cold/hot start & temperature.
"""We can observe from the energy plots that equilibrium is reached faster for
cold starts with a high or low p value. We can also observe that equilibrium
is reached faster for hotter temperatures as expected."""

# Warm start: half spins initially parallel, half initially anti-parallel
run_1d_simulation(N=20, iterations=100, kT=0.1, p=0.5, time=False, plot=True)
run_1d_simulation(N=50, iterations=100, kT=0.5, p=0.5, time=False, plot=True)
run_1d_simulation(N=50, iterations=100, kT=1, p=0.5, time=False, plot=True)

# Cold start: most spins parallel. We could alternatively use a low p
run_1d_simulation(N=20, iterations=100, kT=0.1, p=0.95, time=False, plot=True)
run_1d_simulation(N=50, iterations=100, kT=0.5, p=0.95, time=False, plot=True)
run_1d_simulation(N=50, iterations=100, kT=1, p=0.95, time=False, plot=True)

#%% Question 1b) kT sweep from 0 to 6 with 100 points
"""We observe that the analytical and numerical solutions closely match as
expected. One discrepancy is for entropy where the numerical solution is 
slightly less than the anlaytical solution, possibly due to imprecise 
integration. We can also observe that it gets fully magnetized to 1 or -1 
for low temperatures due to all the spins "locking in". """
run_1d_kT_sweep(N=50, iterations=800, kT=np.linspace(0.01,6,100), p=0.5, time=True, plot=True)

#%% Question 2a) 20x20 lattice - easy to change to 40 or 80 by N parameter
run_2d_simulation(N=20, iterations=100, kT=0.1, p=0.6, time=False, plot=True, frames=[1, 400, 4000, 16000])
run_2d_simulation(N=20, iterations=100, kT=0.5, p=0.6, time=False, plot=True, frames=[1, 400, 4000, 16000])
run_2d_simulation(N=20, iterations=100, kT=1, p=0.6, time=False, plot=True, frames=[1, 400, 4000, 16000])
run_2d_simulation(N=20, iterations=100, kT=3, p=0.6, time=False, plot=True, frames=[1, 400, 4000, 16000])

#%% Question 2b) T/T_c sweep from 0 to 3 with 100 points
run_2d_kT_sweep(N=20, T_Tc=np.linspace(0.01,3,100), p=0.6, time=True, plot=True)
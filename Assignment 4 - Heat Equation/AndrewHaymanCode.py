# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 14:18:19 2022

@author: Andrew Hayman

This program solves the 2D heat equation with space and time discretization
and is setup for parallel computing using MPI. It is meant to be run in a cluster environment
and saves plots and grids at user-specified checkpoints if desired.
"""

"""
Instructions: 
To save grids/plots, please set save_dir to desired saving directory. Otherwise, remove the save_dir 
parameter, or set it to None (line 252)

The slurm_script.sh can run this script with MPI with the command "sbatch slurm_script.sh"

An additional script, main_script.sh has been created to loop through all of the 
desired configurations for timing. This can be run with "./main_script.sh"

Finally, I included the plot_timing_results.py to show my work for the timing
part of the assignment.
"""

#%% Imports
from time import time
start_time = time()
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from functools import wraps
from mpi4py import MPI
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% Matplotlib settings
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.5

#%% Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#%% Timing decorator
"""Place above function to return the time."""
def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te=time()
        dt = te-ts
        return dt
    return wrap

#%% Heat Equation Solver class
class HeatEquationSolver():
    """This class allows users to input settings for solving the 2D heat equation."""

    def __init__(self, config):
        """Store experiment parameters"""
        self.config = config
        self.x_length = config["x_length"]
        self.y_length = config["y_length"]
        self.x_pts = config["x_pts"]
        self.y_pts = config["y_pts"]
        self.D = config["D"]
        self.initializer = config["initializer"]
        self.t_pts = config["t_pts"]
        self.checkpoints = config["checkpoints"]

    @classmethod
    def from_json(cls, config_filename):
        """Allows user to rapidly reload an experiment from a JSON file."""
        with open(config_filename, 'r') as fp:
            config = json.load(fp)
        heat_solver = cls(config)
        return heat_solver

    def initialize(self):
        """Initialize experiment grid. The initial grid is parallelized.
        The grid is split into equally sized horizontal components."""
        self.dx = self.x_length / self.x_pts
        self.dy = self.y_length / self.y_pts
        self.dx2 = self.dx * self.dx
        self.dy2 = self.dy * self.dy
        self.dt = self.dx2 * self.dy2 / (2 * self.D * (self.dx2 + self.dy2))

        x_left = -self.x_length / 2 + self.dx/2
        delta_x = self.x_length/size
        self.x_vals = np.arange(x_left + (rank * delta_x), x_left + ((rank + 1) * delta_x)-self.dx/2, self.dx)
        self.y_vals = np.linspace(-self.y_length / 2, self.y_length / 2, self.y_pts)

        if self.initializer == "custom_initial_function":
            initializer_func = custom_initial_function
        self.u0 = np.zeros((len(self.x_vals), len(self.y_vals)))
        for i in range(len(self.x_vals)):
            for j in range(len(self.y_vals)):
                    self.u0[i][j] = initializer_func(self.x_vals[i], self.y_vals[j])

    @timing
    def solve(self, save_dir=None):
        """Solve heat equation. The two outside grids are given special consideration
        due to boundary conditions and only sharing an edge with one neighboring grid."""
        self.initialize()

        u = self.u0

        chunk = int(self.x_pts / size)

        # Initialize space indices
        if rank==0:
            i_start = 1
        else:
            i_start = 0

        if rank==(size-1):
            i_stop = chunk-1
        else:
            i_stop = chunk

        comm_time = 0
        comp_time = 0

        # Solve heat equation
        for t in range(self.t_pts):

            time1 = time()

            if rank!=0:
                left_nb = comm.recv(source=rank - 1, tag=12)
                comm.send(u[0, 1:-1], dest=rank - 1, tag=11)

            if rank!=size-1:
                comm.send(u[-1, 1:-1], dest=rank + 1, tag=12)
                right_nb = comm.recv(source=rank + 1, tag=11)


            time2 = time()
            comm_time += time2 - time1

            for i in range(i_start,i_stop):
                if i==0:
                    left = left_nb
                else:
                    left = u[i-1, 1:-1]

                if i==(chunk-1):
                    right = right_nb
                else:
                    right = u[i+1, 1:-1]

                u[i, 1:-1] = u[i, 1:-1] + self.D * self.dt * (
                        (u[i, 2:] - 2 * u[i, 1:-1] + u[i, :-2]) / self.dy2
                        + (left - 2 * u[i, 1:-1] + right) / self.dx2)

            time3 = time()
            comp_time += time3 - time2

            # Save grids and plots if applicable
            if t in self.checkpoints and save_dir is not None:

                Path(save_dir).mkdir(parents=True, exist_ok=True)

                full_u = None
                if rank==0:
                    full_u = np.empty((self.x_pts, self.y_pts))
                comm.Gather(u, full_u, root=0)

                if rank==0:
                    save_filename = save_dir + "/heat_eq_" + str(t + 1) + ".npz"
                    np.savez(save_filename, u=full_u, t_step=(t + 1))
                    self.plot(full_u, t, save_dir, save_plots=True)

        print("comm time: %.2f" % (comm_time))
        print("comp time: %.2f" % (comp_time))

        # Save config if applicable
        if rank==0 and save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_filename = save_dir + "/experiment_config.json"
            with open(save_filename, 'w') as fp:
                json.dump(self.config, fp)

    def plot(self, u, t_step, save_dir, save_plots=False, show_plots=False):
        """Contour plot of grid."""
        fig, axes = plt.subplots(1, figsize=(3.2, 2.4))
        im = plt.imshow(u, cmap=plt.get_cmap('hot'), vmin=300, vmax=2000)
        plt.title('{:.1f} ms'.format(t_step * self.dt * 1000))
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="10%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax, ticks=[300,500,750,1000,1250,1500,1750,2000])
        cbar.set_label("K", labelpad=10, rotation=0)
        axes.set_yticks(np.arange(0, 1250, 250))
        axes.set_xticks(np.arange(0,1250,250))
        plt.tight_layout()

        if save_plots:
            plt.savefig(save_dir+"/iter_{}.pdf".format(t_step),
                        format='pdf',
                        #bbox_inches='tight',
                        dpi=200)

        if show_plots:
            plt.show()

        plt.close()

    @staticmethod
    def plot_from_saved_files(save_dir, save_plots=True, show_plots=False):
        """Loads experiment data from directory. Re-initializes heat object
        and uses saved data to generate plots."""
        config_filename = save_dir + "/experiment_config.json"
        heat_solver = HeatEquationSolver.from_json(config_filename)
        heat_solver.initialize()

        for file in os.listdir(save_dir):
            filename = os.fsdecode(file)
            if filename.endswith(".npz"):
                data = np.load(save_dir+"/"+filename)
                u = data['u']
                t_step = data['t_step']
                heat_solver.plot(u, t_step, save_dir, save_plots, show_plots)

#%% Experiment
def custom_initial_function(x, y):
    """Initializes heat based on radius."""
    r = np.sqrt(x ** 2 + y ** 2)
    if r < 5.12:
        return 2000 * (np.cos(4 * r)) ** 4
    else:
        return 300

heat_equation_experiment_config = {"x_length": 20.48,
                                   "y_length": 20.48,
                                   "x_pts": 1024,
                                   "y_pts": 1024,
                                   "D": 4.2,
                                   "initializer": "custom_initial_function",
                                   "t_pts": 1001,
                                   "checkpoints": np.arange(0,1100,100).tolist()
                                   }

save_dir = "./heat_equation_experiment"
heat_solver = HeatEquationSolver(heat_equation_experiment_config)
calc_time = heat_solver.solve(save_dir=save_dir)

print("calc time: %.2f"%(calc_time))
end_time = time()
print("full time: %.2f"%(end_time-start_time))
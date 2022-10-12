#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#%% Matplotlib settings
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.5

#%% Plot function
def plot_cpu_times(save_name, comm_time, comp_time, calc_time, full_time, processes, log_log=False):
    fig, axes = plt.subplots(1, figsize=(3.2, 2.4))

    if(log_log):
        comp_time = np.log(comp_time)
        full_time = np.log(full_time)
        calc_time = np.log(calc_time)
        processes = np.log(processes)

        axes.plot(processes, comp_time, '-o', color='r')
        axes.plot(processes, calc_time, '-o', color='y')
        axes.plot(processes, full_time, '-o', color='g')

        axes.set_xlabel("Log(Number of Processes)")
        axes.set_ylabel("Log(Time) (s)")
        plt.legend(labels=["comp", "full_calc", "full_prog"])
    else:

        axes.plot(processes, comm_time, '-o', color='b')
        axes.plot(processes, comp_time, '-o', color='r')
        axes.plot(processes, calc_time, '-o', color='y')
        axes.plot(processes, full_time, '-o', color='g')

        axes.set_xlabel("Number of Processes")
        axes.set_ylabel("Time (s)")
        plt.legend(labels=["comm", "comp", "full_calc", "full_prog"])

    axes.grid(True, linestyle=':')
    fig.tight_layout()

    if(log_log):
        save_name += "_log"
    plt.savefig("Plots/" + save_name + '.pdf',
                format='pdf',
                dpi=1200,
                bbox_inches='tight')

    plt.show()


#%% Plotting
log_log = False

cac_times_1_full = np.array([36.62, 22.84, 14.26, 11.72])
cac_times_1_calc = np.array([35.14, 21.12, 12.82, 10.17])
cac_times_1_comp = np.array([26.37, 12.90, 6.35, 3.32])
cac_times_1_comm = np.array([0, 6.20, 5.91, 6.42])
cac_processes_1 = np.array([1, 2, 4, 8])
plot_cpu_times("cac_1", cac_times_1_comm, cac_times_1_comp, cac_times_1_calc, cac_times_1_full, cac_processes_1, log_log=log_log)

cac_times_2_full = np.array([23.87, 15.60, 10.55, 8.87])
cac_times_2_calc = np.array([22.06, 14.15, 8.95, 7.24])
cac_times_2_comp = np.array([13.06, 6.72, 3.31, 1.69])
cac_times_2_comm = np.array([7.03, 6.96, 5.46, 5.36])
cac_processes_2 = np.array([2, 4, 8, 16])
plot_cpu_times("cac_2", cac_times_2_comm, cac_times_2_comp, cac_times_2_calc, cac_times_2_full, cac_processes_2, log_log=log_log)
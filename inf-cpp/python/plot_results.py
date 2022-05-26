#!/usr/bin/env python
# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

outdir = 'timing_results'
os.makedirs(outdir, exist_ok=True)

num_threads = [1, 2, 4, 8, 16]
server_types = [0, 1, 2, 3, 4]
steps = ['embedding', 'building', 'filtering', 'gnn', 'total']
upper_limits = [0.005, 0.01, 0.2, 0.02, 0.2]
file_dir = '/global/homes/x/xju/atlas/software/exatrkx-cpp/inf-cpp/build'
# %%
#
# cpu_time = [2.4617, 1.3323, 0.7594, 0.4839, 0.3944] 
# time_local = [0.0426, 0.0762, 0.1863, 0.4890, 0.9583]
# time_ensemble = [0.0545, 0.1007, 0.2033, 0.4010, 0.7928]
# time_ensemble_once = [0.1324, 0.3075, 0.6771, 1.3736, 2.7843]

# # %%
# plt.plot(num_threads, cpu_time, '-o', label='CPU')
# plt.plot(num_threads, time_local, '-o', label='GPU')
# plt.plot(num_threads, time_ensemble, '-o', label='Triton')
# plt.legend()
# plt.xlabel('Number of threads')
# plt.ylabel('Time (s)')
# plt.xticks(num_threads)
# plt.savefig("time_vs_threads.png")

# %%
###############################################################################
#### Impact of threading on different models
###############################################################################
time_s4 = [pd.read_csv(file_dir + '/time_t{:d}_s4.csv'.format(i)) for i in num_threads]
# %%
for idx in range(len(num_threads)):
    array = time_s4[idx].total.to_numpy()[num_threads[idx]:]
    mean, std = np.mean(array), np.std(array)
    plt.hist(array,
        bins=50, label='t{:02}: {:.4f}'.format(num_threads[idx], mean),
        range=(0, 1),
        histtype='step', density=True, lw=2)

plt.xlabel("Time per event [s]")
plt.ylabel("Events")
plt.title("Ensemble Model")
plt.legend()
plt.savefig(os.path.join(outdir, "time_ensemble_threading.png"))
# %%
time_s0 = [pd.read_csv(file_dir + '/time_t{:d}_s0.csv'.format(i)) for i in num_threads]
# %%
for idx in range(len(num_threads)):
    array = time_s0[idx].total.to_numpy()[num_threads[idx]:]
    mean, std = np.mean(array), np.std(array)
    plt.hist(array,
        bins=50, label='t{:02}: {:.4f}'.format(num_threads[idx], mean),
        range=(0, 1),
        histtype='step', density=True, lw=2)

plt.xlabel("Time per event [s]")
plt.ylabel("Events")
plt.title("Local Model")
plt.legend()
plt.savefig(os.path.join(outdir, "time_local_threading.png"))

# %%
###############################################################################
## timing information for different server options with one thread
## enumerate on each step
###############################################################################
time_t1 = [pd.read_csv(file_dir + '/time_t1_s{:d}.csv'.format(i)) for i in server_types]
# %%
for istep,step in enumerate(steps):
    upper = upper_limits[istep]
    plt.cla()

    tot_options = len(server_types)-1 if istep < 4 else len(server_types)
    for idx in range(tot_options):
        array = time_t1[idx][step].to_numpy()[1:]
        array = array[array < upper]
        mean, std = np.mean(array), np.std(array)
        plt.hist(array,
            bins=50, label='s{:02}: {:.4f}'.format(server_types[idx], mean),
            range=(0, upper),
            histtype='step', density=True, lw=2)

    plt.xlabel("{} time [s]".format(step))
    plt.ylabel("Events")
    plt.legend()
    plt.savefig(os.path.join(outdir, "time_onethread_{}.png".format(step)),
        bbox_inches='tight', dpi=300)

# %%

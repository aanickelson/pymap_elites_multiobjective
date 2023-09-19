import numpy as np
import os
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # File name
    base_fname = '/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data/554_20230914_175010/'
    # Get subfile names
    bh_files = ['200000_battery_distance_run0', '200000_battery_type combo_run0', '200000_battery_type sep_run0',
                '200000_full act_run0', '200000_type combo_run0', '200000_type sep_run0', '200000_v or e_run0']
    sub_files = ['top_20230915_124314_ob']
    n_gen = 500
    for bhf in bh_files:
        for s in sub_files:
            fname = f"{base_fname}{bhf}/{s}"
            file_data = list(os.walk(fname))[0][2]
            all_data = np.zeros((len(file_data), n_gen))
            for i, f in enumerate(file_data):
                # Import data from 5 files
                if "NOTES" in f:
                    continue
                all_data[i] = np.load(f"{fname}/{f}")
            avgs = np.average(all_data, axis=0)
            stds = np.std(all_data, axis=0)
            print(avgs[99], stds[99])
            x = np.array([i for i in range(len(stds))])
            plt.clf()
            plt.title(s)
            plt.plot(x, avgs)
            plt.fill_between(x, avgs-stds, avgs+stds, alpha=0.2)
            plt.show()

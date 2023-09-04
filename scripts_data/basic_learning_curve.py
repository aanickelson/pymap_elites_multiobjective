import numpy as np
import os
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # File name
    base_fname = '/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data/536_20230903_124548/111101_run0/'
    # Get subfile names
    sub_files = ['top_20230904_134004_ob', 'top_20230904_135614_b', 'top_20230904_141055_o']
    n_gen = 300
    for s in sub_files:
        fname = base_fname + s
        file_data = list(os.walk(fname))[0][2]
        all_data = np.zeros((len(file_data), n_gen))
        for i, f in enumerate(file_data):
            # Import data from 5 files
            if "NOTES" in f:
                continue
            all_data[i] = np.load(f"{fname}/{f}")
        avgs = np.average(all_data, axis=0)
        stds = np.std(all_data, axis=0)
        print(avgs[-1], stds[-1])
        x = np.array([i for i in range(len(stds))])
        plt.clf()
        plt.title(s)
        plt.plot(x, avgs)
        plt.fill_between(x, avgs-stds, avgs+stds, alpha=0.2)
        plt.show()

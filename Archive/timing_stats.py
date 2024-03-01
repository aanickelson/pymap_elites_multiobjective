import re

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import os
from pymap_elites_multiobjective.scripts_data.plot_pareto import get_area


def load_data(filename):
    data = np.loadtxt(filename)
    return data


if __name__ == '__main__':

    # Change these parameters to run the script
    dates = ['507_20230523_180028']  # Change the dates to match the date code on the data set(s) you want to use
    param_sets = ['010', '349']

    time_stats_dict = {pname: [] for pname in param_sets}

    for date in dates:
        rootdir = os.path.join(os.getcwd(), 'data', date)
        # Walk through all the files in the given directory
        files = list(os.walk(rootdir))[0][2]
        for file in files:
            p_num = re.split('_|/', file)[0]
            d = load_data(os.path.join(rootdir, file))
            time_stats_dict[p_num].append(d)

    for key, val in time_stats_dict.items():
        print(key, np.mean(val))

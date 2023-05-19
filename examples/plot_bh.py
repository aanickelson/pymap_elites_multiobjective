import re

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import os


def load_data(filename):
    data = np.loadtxt(filename)
    return data


def process(c_vals, p_vals):
    centroids, counts = np.unique(p_vals, return_counts=True, axis=0)
    return len(counts) / c_vals.shape[0]


if __name__ == '__main__':

    # Change these parameters to run the script
    n_files = 20  # Need this in order to make sure the number of data points is consistent for the area plot
    dates = ['10_to_13', '20230518_163348']  # Change the dates to match the date code on the data set(s) you want to use
    param_sets = ['010', '011', '012', '013', '245', '345']
    final_num = 20000

    all_dates = '_'.join(dates)
    # Filename setup
    graphs_fname = os.path.join(os.getcwd(), 'graphs_' + all_dates)
    try:
        os.mkdir(graphs_fname)
    except FileExistsError:
        pass

    params_dict = {pname: [] for pname in param_sets}

    for date in dates:
        if 'to' in date:
            num_to_grab = 3
        else:
            num_to_grab = 2
        rootdir = os.path.join(os.getcwd(), date)
        # Walk through all the files in the given directory
        for sub, _, files in os.walk(rootdir):
            # If there are no files, move on to the next item
            if not files:
                continue

            c_ = False
            p_ = False

            # Get the name of the sub-directory
            params_name = re.split('_|/', sub)[-num_to_grab]

            for file in files:
                full_fpath = os.path.join(sub, file)
                if 'centroids' in file:
                    c_data = load_data(full_fpath)
                    c_ = True
                elif str(final_num) in file:
                    p_data = load_data(full_fpath)[:, 2:7]
                    p_ = True
                else:
                    continue

            if not c_ or not p_:
                print(f"not enough data in {sub}")
                continue

            pct_filled = process(c_data, p_data)
            params_dict[params_name].append(pct_filled)

    print(params_dict)

    for key, value in params_dict.items():
        print(key, np.mean(value))

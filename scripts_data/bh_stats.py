import re

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import os
from pymap_elites_multiobjective.scripts_data.plot_pareto import get_area


def load_data(filename):
    data = np.loadtxt(filename)
    return data


def process(c_vals, p_vals):
    centroids, counts = np.unique(p_vals, return_counts=True, axis=0)
    return len(counts) / c_vals.shape[0]


if __name__ == '__main__':

    # Change these parameters to run the script
    n_files = 20  # Need this in order to make sure the number of data points is consistent for the area plot
    dates = ['004_20230509_182108', '006_20230518_161237']  # Change the dates to match the date code on the data set(s) you want to use
    param_sets = ['239', '249', '349']


    # param_sets = ['231', '233', '235', '237', '239',
    #               '241', '243', '245', '247', '249',
    #               '341', '343', '345', '347', '349']
    # '010', '011', '013',
    #                   '021', '023',
    #                   '031', '033',
    #                   '041', '043',
    #                   '121', '123',
    #                   '141', '143',
    final_num = 200000

    bh_stats_dict = {pname: [] for pname in param_sets}
    area_stats_dict = {pname: [] for pname in param_sets}

    for date in dates:
        if 'to' in date:
            num_to_grab = 3
        else:
            num_to_grab = 2
        rootdir = os.path.join(os.getcwd(), 'data', date)
        # Walk through all the files in the given directory
        for sub, _, files in os.walk(rootdir):
            # If there are no files, move on to the next item
            if not files:
                continue

            params_name = re.split('_|/', sub)[-num_to_grab]
            if not params_name in bh_stats_dict:
                print(f'## SKIPPING {sub}')
                continue

            print(f'processing {sub}')


            c_ = False
            p_ = False

            # Get the name of the sub-directory

            for file in files:
                full_fpath = os.path.join(sub, file)
                if 'centroids' in file:
                    c_data = load_data(full_fpath)
                    c_ = True
                    print(file)

                elif str(final_num) in file:
                    policy_data = load_data(full_fpath)
                    p_data = policy_data[:, 2:7]
                    pareto_area = get_area(policy_data[:, 0], policy_data[:, 1])
                    p_ = True
                    print(file)
                else:
                    continue

            if not c_ or not p_:
                print(f"not enough data in {sub}")
                continue
            pct_filled = process(c_data, p_data)
            if params_name in bh_stats_dict:
                bh_stats_dict[params_name].append(pct_filled)
                area_stats_dict[params_name].append(pareto_area)
            else:
                print(f'{params_name} not in dictionary')

    print('########## BEHAVIOR ##########')
    print(bh_stats_dict)

    for key, value in bh_stats_dict.items():
        print(key, np.mean(value))

    print('########## PARETO ##########')
    print(area_stats_dict)

    for k, v in area_stats_dict.items():
        print(k, np.mean(v))

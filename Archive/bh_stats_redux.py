import re

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import os
from pymap_elites_multiobjective.scripts_data.often_used import is_pareto_efficient_simple
import pygmo
import itertools

##############################
# This block is for file i/o #
##############################

def get_file_info(dates, cwd=None):
    if not cwd:
        cwd = os.getcwd()

    files_to_use = []
    for date in dates:
        root_dir = os.path.join(cwd, 'data', date)
        sub_dirs = list(os.walk(root_dir))[0][1]
        for s in sub_dirs:
            pol = None
            cent = None
            sub = os.path.join(root_dir, s)
            files = os.listdir(sub)
            # If there are no files, move on to the next item
            if not files:
                continue
            # This block gets the file numbers of all the archives
            for file in files:
                if 'centroids' in file:
                    cent = os.path.join(sub, file)
                if 'archive_200000' in file:
                    pol = os.path.join(sub)
            if not pol or not cent:
                continue

            # Get the name of the sub-directory
            params_name = sub.split('/')[-1]

            files_to_use.append([date, params_name, pol, cent])

    return files_to_use


def load_data(filename):
    data = np.loadtxt(filename)
    return data

def process_centroids(c_vals, p_vals):
    centroids, counts = np.unique(p_vals, return_counts=True, axis=0)
    return len(counts) / c_vals.shape[0]

##############################################
# This block is for calculating pareto areas #
##############################################

def get_area(xy):
    is_eff = is_pareto_efficient_simple(xy)
    hyper = get_hypervolume(xy[is_eff])
    return hyper

def get_hypervolume(xy_vals):
    xy = -1 * np.array(xy_vals)
    hv = pygmo.hypervolume(xy)
    return hv.compute([0.0]*2)  # returns the exclusive volume by point 0
    # hv.least_contributor(r=[0, ])  # returns the index of the least contributor




if __name__ == '__main__':

    # Change these parameters to run the script
    n_files = 20  # Need this in order to make sure the number of data points is consistent for the area plot

    basedir_qd = os.getcwd()
    dates = ['507_20230523_180028', '003_20230505_171536', '004_20230509_182108', '007_20230522_123227']
    # params_set = ['010', '231', '233', '235', '237', '239',
    #               '241', '243', '245', '247', '249',
    #               '341', '343', '345', '347', '349']
    params_set = ['010', '239','249','349']
    evols = [(i + 1) * 10000 for i in range(n_files)]

    files = get_file_info(dates, os.getcwd())

    data_dict = {p:[[],[]] for p in params_set}

    # Walk through all the files in the given directory
    for date, params_name, arch_f, cent_f in files:
        # Pulls the parameter file number
        p_num = params_name[:3]
        if p_num not in params_set:
            continue
        bh_size = 5
        if '010' in p_num:
            bh_size = 6

        centroids = load_data(cent_f)
        areas = []
        pcts = []
        for evo in evols:
            try:
                policies = load_data(os.path.join(arch_f, f'archive_{evo}.dat'))
            except FileNotFoundError:
                print(f'NOT FOUND: {arch_f}, {evo} ')
                continue
            pol_obj = policies[:, :2]
            pol_bh = policies[:, 2:2+bh_size]
            area = get_area(pol_obj)
            cent_pct = process_centroids(centroids, pol_bh)
            areas.append(area)
            pcts.append(cent_pct)
        data_dict[p_num][0].append(areas)
        data_dict[p_num][1].append(pcts)
        print(date, params_name, areas[-1], pcts[-1])

    for key, item in data_dict.items():
        print(key)
        print(np.mean(item[0], axis=0)[-1])
        print(np.mean(item[1], axis=0)[-1])

        # print(f'{p_num}, {area:.04f}, {cent_pct:.04f}',)  # '         ', params_name, date)
        # if abs(area - 1) < 0.005:
        #     print(arch_f)
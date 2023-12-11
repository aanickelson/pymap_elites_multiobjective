import re

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import os
from pymap_elites_multiobjective.scripts_data.often_used import is_pareto_efficient_simple
import pygmo
# import platypus

##############################
# This block is for file i/o #
##############################

def file_setup(f_dates, pre='', cwd=None):
    fnums = [re.split('_|/', d)[0] for d in f_dates]

    all_fnums = '_'.join(fnums)

    if not cwd:
        cwd = os.getcwd()
    # Filename setup
    graphs_dir = os.path.join(cwd, 'graphs')
    if not os.path.exists(graphs_dir):
        os.mkdir(graphs_dir)

    graphs_f = os.path.join(graphs_dir, f'{pre}graphs_{all_fnums}')
    if not os.path.exists(graphs_f):
        os.mkdir(graphs_f)

        text_f = os.path.join(graphs_f, 'NOTES.txt')
        with open(text_f, 'w') as f:
            f.write('Dates: ')
            for dt in f_dates:
                f.write(f'{dt}, ')
            f.write('\n')

    return graphs_f


def get_file_info(dates, a_or_f, cwd=None):
    if not cwd:
        cwd = os.getcwd()

    files_to_use = []
    for date in dates:
        root_dir = os.path.join(cwd, date)
        sub_dirs = list(os.walk(root_dir))[0][1]
        for s in sub_dirs:
            sub = os.path.join(root_dir, s)
            files = os.listdir(sub)
            # If there are no files, move on to the next item
            if not files:
                continue
            # This block gets the file numbers of all the archives
            fnums = []
            for file in files:
                # 'archive' is what all the data is saved to
                if a_or_f not in file:
                    continue
                try:
                    # Find all file numbers (the final number appended, which is the number of policies tested so far)
                    fnums.append(int(re.findall(r'\d+', file)[0]))
                except IndexError:
                    print(file, "index error")
                    continue
            # Sort all file numbers
            fnums.sort()
            # Get the name of the sub-directory
            params_name = sub.split('/')[-1]
            files_to_use.append([sub, date, params_name, fnums])

    return files_to_use


def load_data(filename, n_objective, plot_obj):
    data = np.loadtxt(filename)
    xvals = data[:, plot_obj[0]]
    yvals = data[:, plot_obj[1]]
    xyvals = data[:, 0:n_objective]
    return xvals, yvals, xyvals

def load_centroids(filename):
    data = np.loadtxt(filename)
    return data

def process_centroids(c_vals, p_vals):
    centroids, counts = np.unique(p_vals, return_counts=True, axis=0)
    return len(counts) / c_vals.shape[0]

##############################################
# This block is for calculating pareto areas #
##############################################

def get_area(xy_v, orig):
    xy = np.array(xy_v)
    hv = pygmo.hypervolume(xy)
    # try:
    return hv.compute(orig)  # returns the exclusive volume by point 0
    # except ValueError:
    #     print(xy_v, orig)

def get_areas_in_sub(sub, fnms, pnum, plot_sc, dt, paramsnm, graphs_f, f_types, a_or_f, n_objectives, obj_to_plot, origin=None):
    if not origin:
        origin = [0.0]*n_objectives
    areas = []
    for evo in fnms:
        if evo == 0:
            continue

        if a_or_f == 'archive_':
            ext = '.dat'
        elif a_or_f == 'fits':
            ext = '.npy'
        else:
            print('something went wrong in the areas')
            return

        fname = os.path.join(sub, f'{a_or_f}{evo}{ext}')
        if not os.path.exists(fname):
            continue
        x, y, xy = load_data(fname, n_objectives, obj_to_plot)
        is_eff = is_pareto_efficient_simple(xy)

        # if evo == 1998:
        #     evo = 2000
        # if a_or_f == 'fits':
        #     evo *= 100
        if evo == fnms[-1] and plot_sc:
            curve_area = plot_pareto_scatter(x, y, xy, is_eff, f'{pnum}_{evo}',
                                             f'{pnum}_{dt}_{paramsnm}', graphs_f, f_types, origin)
        else:
            curve_area = get_area(xy[is_eff], origin)

        areas.append(curve_area)

    return areas, x[is_eff], y[is_eff]


def process(data):
    from scipy.stats import sem
    mu = np.mean(data, axis=0)
    ste = sem(data, axis=0)
    # mu = data[0]
    # ste = data[0]
    return mu, ste


##############################
# This block is for plotting #
##############################

def plot_pareto_scatter(x, y, xy, iseff, graph_title, fname, graph_dir, filetypes, orgn):
    dirname = os.path.join(graph_dir, 'pareto')
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    plt.clf()
    max_vals = [max(x), max(y)]
    min_vals = [min(x), min(y)]
    # max_vals = [8.3, 8.3]
    # plt_max = 2.3
    # plt_max = 3.3
    # plt_max = 90
    if min_vals[0] < 0 and min_vals[1] < 0:
        plt.xlim([min_vals[0] * 1.05, 0.1])
        plt.ylim([min_vals[1] * 1.05, 0.1])
    else:
        plt.xlim([-0.1, max_vals[0] * 1.05])
        plt.ylim([-0.1, max_vals[1] * 1.05])
    plt.scatter(x, y, c='red')
    plt.scatter(x[iseff], y[iseff], c="blue")
    curve_area = get_area(xy[iseff], orgn)
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.title(f"{graph_title} AREA: {curve_area:.03f}")
    for ext in filetypes:
        plt.savefig(os.path.join(dirname, fname + ext))
    return curve_area


def plot_areas(evos, data_and_names, dirname, graphs_dir_fname, filetypes):
    print('We made it to plotting')
    plt.clf()
    # plt.ylim([-0.1, 2.1])
    evos = np.array(evos)
    text_f = os.path.join(graphs_dir_fname, f'NOTES_{dirname}_means.txt')
    mrks = ['.', '*', 'o', 'v', 'P', 'D', '>', '1', '2', '3', '4', '.', '*', 'o', 'v', 'P', 'D', '>', '1']
    with open(text_f, 'w') as f:
        f.write(f'Final Means, {dirname}\n')
    mrk_n = 0
    max_mean = 0
    for nm, data in data_and_names.items():
        if not data:
            continue
        means, sterr = process(data)
        if max(means) > max_mean:
            max_mean = max(means) + 0.2

        with open(text_f, 'a') as f:
            f.write(f'{nm} & {means[-1]} & {sterr[-1]} \n')

        # these are print statements if you want to print & combine data across machines
        # It's far easier this way than trying to migrate 2-3GB of raw data across machines.
        # print(nm)
        # print(repr(means))
        # print(repr(sterr))
        plt.plot(evos, means, marker=mrks[mrk_n], label=nm)
        mrk_n += 1
        plt.fill_between(evos, means-sterr, means+sterr, alpha=0.3)
    plt.ylim([-0.1, max_mean * 1.05])
    # plt.ylim([-0.1, 3.1])
    # plt.title(f"{dirname}")
    plt.xlabel('Number of Policies Tested')
    plt.ylabel('Hypervolume of resulting Pareto front')
    plt.legend()
    try:
        os.mkdir(graphs_dir_fname)
    except FileExistsError:
        pass
    for ext in filetypes:
        plt.savefig(os.path.join(graphs_dir_fname, dirname + ext))


if __name__ == '__main__':

    # Change these parameters to run the script

    ftypes = ['.png']  #, '.svg']   # What file type(s) do you want for the plots  '.svg',

    plot_scatters = True   # Do you want to plot the scatter plots of the objective space for each data set
    n_files = 10  # Need this in order to make sure the number of data points is consistent for the area plot

    # If you don't define this, it will use the current working directory of this file
    dates_qd = ['005_20231211_160926']
    gym_dir_name = 'mountain'
    basedir_qd = os.path.join(os.getcwd(), 'data_gym', gym_dir_name)
    files_info = [[dates_qd, basedir_qd, 'archive_']]

    graphs_fname = file_setup(dates_qd, cwd=basedir_qd)
    evols = [(i + 1) * 1000 for i in range(n_files)]
    n_obj = 4
    plot_obj_idx = [1,3]
    param_num = 0
    param_nms = ['avg act', 'avg st', 'fin act', 'fin st', 'min avg max act']
    param_sets = ['000']
    data_and_nm = {p: [] for p in param_nms}
    plot_fname = 'lander'  # What domain is being tested
    orig = [0.5]*n_obj
    # orig = [-110., -35., -1., -20.]

    from time import time

    for dates, basedir, arch_or_fits in files_info:
        files = get_file_info(dates, arch_or_fits, basedir)

        # Walk through all the files in the given directory
        for sub, date, params_name, fnums in files:
            # Pulls the parameter file number
            # p_num = params_name[:3]
            p_num = params_name.split('_')[0]
            bh_name = params_name.split('_')[1]
            if not p_num in param_sets:
                # print(f'Did not save data for {params_name} in {sub}')
                continue
            if len(fnums) < n_files:
                continue

            # This block goes through each fil609047338827017e, gets the data, finds the pareto front, gets the area, then saves the area

            start = time()
            areas, x_p, y_p = get_areas_in_sub(sub, fnums, p_num, plot_scatters, date, params_name, graphs_fname, ftypes, arch_or_fits, n_obj, plot_obj_idx, origin=orig)[:n_files]
            end_time = time() - start
            print(f'{bh_name}, {areas[-1]}')
            # areas = [a / 10000 for a in areas]
            # print(f"{end_time}")
            if len(areas) < n_files:
                print(f'Not enough files with data for {params_name} in {sub}')
                continue
            elif len(areas) > n_files:
                areas = areas[:n_files]

            try:
                data_and_nm[bh_name].append(areas)
            except KeyError:
                print(f'Did not save data for {params_name} in {sub}')
                continue

    # Plot the areas data for all parameters on one plot to compare
    plot_areas(evols, data_and_nm, plot_fname, graphs_fname, ftypes)

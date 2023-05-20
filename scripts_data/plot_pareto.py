import re

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import os
from pymap_elites_multiobjective.scripts_data.often_used import is_pareto_efficient_simple

##############################
# This block is for file i/o #
##############################

def file_setup(f_dates):
    fnums = [re.split('_|/', d)[0] for d in f_dates]

    all_fnums = '_'.join(fnums)
    # Filename setup
    graphs_dir = os.path.join(os.getcwd(), 'data', 'graphs')
    if not os.path.exists(graphs_dir):
        os.mkdir(graphs_dir)

    graphs_f = os.path.join(graphs_dir, 'graphs_' + all_fnums)
    if not os.path.exists(graphs_f):
        os.mkdir(graphs_f)

    text_f = os.path.join(graphs_f, 'NOTES.txt')
    with open(text_f, 'w') as f:
        f.write('Dates: ')
        for dt in f_dates:
            f.write(f'{dt}, ')

    return graphs_f


def get_file_info(dates):

    files_to_use = []
    for date in dates:
        root_dir = os.path.join(os.getcwd(), 'data', date)
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
                if 'archive' not in file:
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


def load_data(filename):
    data = np.loadtxt(filename)
    xvals = data[:, 0]
    yvals = data[:, 1]
    xyvals = data[:, 0:2]
    return xvals, yvals, xyvals


##############################################
# This block is for calculating pareto areas #
##############################################

def get_area(x, y):
    xy = list(zip(x, y))
    # Add the origin
    xy.append((0, 0))
    # Add the x-axis intercept
    xy.append((max(x), 0))
    # Add the y-axis intercept
    xy.append((0, max(y)))

    unique_xy = np.unique(xy, axis=0)

    pgon = Polygon(unique_xy)

    return pgon.area


def get_areas_in_sub(sub, fnms, pnum, plot_sc, dt, paramsnm, graphs_f, f_types):
    areas = []
    for evo in fnms:
        fname = os.path.join(sub, f'archive_{evo}.dat')
        if not os.path.exists(fname):
            continue
        x, y, xy = load_data(fname)
        is_eff = is_pareto_efficient_simple(xy)
        if evo == fnums[-1] and plot_sc:
            curve_area = plot_pareto_scatter(x, y, is_eff, f'{pnum}_{evo}',
                                             f'{dt}_{paramsnm}_{evo}', graphs_f, f_types)
        else:
            curve_area = get_area(x[is_eff], y[is_eff])
        areas.append(curve_area)

    return areas


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

def plot_pareto_scatter(x, y, iseff, graph_title, fname, graph_dir, filetypes):
    dirname = os.path.join(graph_dir, 'pareto')
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    plt.clf()
    max_vals = [max(x), max(y)]
    plt_max = 2.3
    # plt_max = 90
    plt.xlim([-0.1, plt_max])
    plt.ylim([-0.1, plt_max])
    plt.scatter(x, y, c='red')
    plt.scatter(x[iseff], y[iseff], c="blue")
    curve_area = get_area(x[iseff], y[iseff])
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.title(f"{graph_title} AREA: {curve_area:.03f}")
    for ext in filetypes:
        plt.savefig(os.path.join(dirname, fname + ext))
    return curve_area


def plot_areas(evos, data_and_names, dirname, area_fname, filetypes):
    print('We made it to plotting')
    plt.clf()
    plt.ylim([-0.1, 2.1])
    evos = np.array(evos)
    for [data, nm] in data_and_names:
        if not data:
            continue
        means, sterr = process(data)
        # these are print statements if you want to print & combine data across machines
        # It's far easier this way than trying to migrate 2-3GB of raw data across machines.
        # print(nm)
        # print(repr(means))
        # print(repr(sterr))
        plt.plot(evos, means)
        plt.fill_between(evos, means-sterr, means+sterr, alpha=0.5, label=nm)
    plt.title(f"{dirname}")
    plt.xlabel('Number of Policies Tested')
    plt.ylabel('Hypervolume of resulting Pareto front')
    plt.legend()
    try:
        os.mkdir(area_fname)
    except FileExistsError:
        pass
    for ext in filetypes:
        plt.savefig(os.path.join(area_fname, dirname + ext))


if __name__ == '__main__':

    # Change these parameters to run the script
    n_files = 20  # Need this in order to make sure the number of data points is consistent for the area plot
    dates = ['005_20230518_174201', '004_20230518_163348']  # Change the dates to match the date code on the data set(s) you want to use

    ftypes = ['.svg', '.png']   # What file type(s) do you want for the plots

    plot_scatters = True   # Do you want to plot the scatter plots of the objective space for each data set

    # FOR PARAMETER FILE NAME CODES -- see __NOTES.txt in the parameters directory

    # all_sets is a little wonky, I'll admit.
    # Each set is [[param file numbers], [param names for plot], 'graph title']
    # Param names provides the name of each parameter being compared. Should line up with the files
    # In this example, the names are consistent across all the plots, but they won't always be depending on what you want to run
    nms = ['0 cf', '1 cf', '5 cf', '9 cf']

    all_sets = [[['010', '231', '235',  '239'], nms, 'Num Counterfactuals, Static'],
                [['010', '241', '245', '249'], nms, 'Num Counterfactuals, Move, no POI'],
                [['010', '341', '345', '349'], nms, 'Num Counterfactuals, Move, POI']]

    # all_sets = [[['010', '239', '249', '349'], ['0 cf', 'Static', 'Move', 'POI'], '9 Counterfactuals']]

    # If you want to collect multiple parameter sets into one set, use this style
    # batch_0cf = ['010']
    # batch_no_3 = ['013', '033']
    # batch_move_3 = ['023', '043']
    # batch_poi_3 = ['123', '143']
    # nms = ['No cf', 'Static cf', 'Moving cf', 'Task cf']
    # all_sets = [[[batch_0cf, batch_no_3, batch_move_3, batch_poi_3], nms, '3 Counterfactuals']]

    # You shouldn't need to change anything beyond here
    # ---------------------------------------------------------

    files = get_file_info(dates)

    for param_sets, param_names, nm in all_sets:
        plot_fname = f'{nm}'  # What domain is being tested

        graphs_fname = file_setup(dates)

        evols = [(i + 1) * 1000 for i in range(n_files)]
        data_and_nm = [[[], p] for p in param_names]

        # Walk through all the files in the given directory
        for sub, date, params_name, fnums in files:
            print(params_name)
            # Pulls the parameter file number
            p_num = params_name[:3]
            # Save the areas to the appropriate parameter set
            it_worked = False
            for i, p_name in enumerate(param_sets):
                if p_num in p_name:
                    # This block goes through each file, gets the data, finds the pareto front, gets the area, then saves the area
                    areas = get_areas_in_sub(sub, fnums, p_num, plot_scatters, date, params_name, graphs_fname, ftypes)
                    if len(areas) < n_files:
                        continue

                    data_and_nm[i][0].append(areas)
                    it_worked = True
            if not it_worked:
                print('This file will not be included in the final graph')

        # Plot the areas data for all parameters on one plot to compare
        plot_areas(evols, data_and_nm, plot_fname, graphs_fname, ftypes)

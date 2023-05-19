import re

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import os


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


def is_pareto_efficient_simple(xyvals):
    """
    Find the pareto-efficient points
    This function copied from here: https://stackoverflow.com/a/40239615
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    costs = np.array(xyvals)
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)  # Keep any point with a lower cost
            # The two lines below this capture points that are equal to the current compared point
            # Without these two lines, it will only keep one of each pareto point
            # E.g. if there are two policies that both get [4,0], only the first will be kept. That's bad.
            eff_add = np.all(costs == c, axis=1)
            is_efficient += eff_add
            is_efficient[i] = True  # And keep self
    return is_efficient


def load_data(filename):
    data = np.loadtxt(filename)
    xvals = data[:, 0]
    yvals = data[:, 1]
    xyvals = data[:, 0:2]
    return xvals, yvals, xyvals


def process(data):
    from scipy.stats import sem
    mu = np.mean(data, axis=0)
    ste = sem(data, axis=0)
    # mu = data[0]
    # ste = data[0]
    return mu, ste


def plot_pareto_scatter(x, y, iseff, graph_title, fname, dirname, filetypes):
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
    dates = ['20230505_171536', '20230509_182108']  # Change the dates to match the date code on the data set(s) you want to use
    ftypes = ['.svg', '.png']   # What file type(s) do you want for the plots
    plot_scatters = False   # Do you want to plot the scatter plots of the objective space for each data set

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
    for param_sets, param_names, nm in all_sets:
        plot_fname = f'{nm}'  # What domain is being tested

        all_dates = '_'.join(dates)
        # Filename setup
        graphs_fname = os.path.join(os.getcwd(), 'graphs_' + all_dates)
        try:
            os.mkdir(graphs_fname)
        except FileExistsError:
            pass

        evols = [(i + 1) * 10000 for i in range(n_files)]
        data_and_nm = [[[], p] for p in param_names]

        for date in dates:
            rootdir = os.path.join(os.getcwd(), date)
            # Walk through all the files in the given directory
            for sub, _, files in os.walk(rootdir):
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
                print(sub)

                # Get the name of the sub-directory
                params_name = sub.split('/')[-1]

                # Pulls the parameter file number
                p_num = params_name[:3]
                # Save the areas to the appropriate parameter set
                it_worked = False
                for i, p_name in enumerate(param_sets):
                    if p_num in p_name:
                        # This block goes through each file, gets the data, finds the pareto front, gets the area, then saves the area
                        areas = []
                        for evo in fnums:
                            fname = os.path.join(rootdir, sub, f'archive_{evo}.dat')
                            try:
                                x, y, xy = load_data(fname)
                            except FileNotFoundError:
                                continue
                            is_eff = is_pareto_efficient_simple(xy)
                            if evo == fnums[-1] and plot_scatters:
                                curve_area = plot_pareto_scatter(x, y, is_eff, f'{p_num}_{evo}',
                                                                 f'{date}_{params_name}_{evo}', graphs_fname, ftypes)
                            else:
                                curve_area = get_area(x[is_eff], y[is_eff])
                            areas.append(curve_area)
                        if len(areas) < n_files:
                            continue

                        data_and_nm[i][0].append(areas)
                        it_worked = True
                if not it_worked:
                    print('This file will not be included in the final graph')

        # Plot the areas data for all parameters on one plot to compare
        plot_areas(evols, data_and_nm, plot_fname, graphs_fname, ftypes)

import re

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import os


def plot_it(x, y, iseff, fname, dirname):
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
    plt.title(f"{fname} AREA: {curve_area}")
    plt.savefig(os.path.join(dirname, fname + '.png'))
    return curve_area


def get_area(x, y):
    try:
        xy = list(zip(x, y))
        # Add the origin
        xy.append((0, 0))
        # Add the x-axis intercept
        xy.append((max(x), 0))
        # Add the y-axis intercept
        xy.append((0, max(y)))

        unique_xy = np.unique(xy, axis=0)

        pgon = Polygon(unique_xy)
    except RuntimeWarning:
        print(x)
        print(y)
        exit(1)
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
    try:
        mu = np.mean(data, axis=0)
        ste = sem(data, axis=0)
    except RuntimeWarning:
        print(data)
    # mu = data[0]
    # ste = data[0]
    return mu, ste


def plot_areas(evos, data_and_names, dirname, area_fname):
    print('We made it to plotting')
    plt.clf()
    evos = np.array(evos)
    for [data, nm] in data_and_names:
        if not data:
            continue
        try:
            means, sterr = process(data)
            plt.plot(evos, means)
        except RuntimeWarning:
            print("'tis here, mlord")
            continue
        plt.fill_between(evos, means-sterr, means+sterr, alpha=0.5, label=nm)
    plt.title(f"{dirname} areas")
    plt.legend()
    try:
        os.mkdir(area_fname)
    except FileExistsError:
        pass
    plt.savefig(os.path.join(area_fname, dirname + '.png'))


if __name__ == '__main__':

    # Change these parameters to run the script
    n_files = 20  # Need this in order to make sure the number of data points is consistent for the area plot
    domain_name = 'AIC'  # What domain is being tested
    dates = ['20230417_170533', '20230412_134629', '20230418_160058', '20230419_163935', '20230420_164143']  # ['20230417_170533', '20230412_134629', '20230418_160058', ]  # Date stamp of data folder
    param_set = ['003', '004']  # Distinguishing factor in the filenames of parameter you want to test (e.g. diff param files, different selection types, etc)
    all_dates = '_'.join(dates)
    # Filename setup
    # rootdir = os.path.join(os.getcwd(), '_graphs' + all_dates)
    # os.mkdir(rootdir)
    graphs_fname = os.path.join(os.getcwd(), 'graphs_' + all_dates)
    os.mkdir(graphs_fname)

    pareto = 'no'  # This is legacy, but currently still necessary
    evols = [(i + 1) * 10000 for i in range(n_files)]
    data_and_nm = [[[], p] for p in param_set]

    for date in dates:
        rootdir = os.path.join(os.getcwd(), date)
        # area_fname = os.path.join(rootdir, 'graphs')
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

            # This block goes through each file, gets the data, finds the pareto front, gets the area, then saves the area
            areas = []
            for evo in fnums:
                fname = os.path.join(rootdir, sub, f'archive_{evo}.dat')
                try:
                    x, y, xy = load_data(fname)
                except FileNotFoundError:
                    continue
                is_eff = is_pareto_efficient_simple(xy)
                # Use this line if you want to plot the evoloution of the pareto fronts over time
                if evo == fnums[-1]:
                    curve_area = plot_it(x, y, is_eff, f'{date}_{params_name}_{pareto}_{evo}', graphs_fname)
                # Use this line if you only want the areas for the final plot, and not the individual pareto plots
                else:
                    curve_area = get_area(x[is_eff], y[is_eff])
                areas.append(curve_area)
            if len(areas) < n_files:
                continue

            # Save the areas to the appropriate parameter set
            it_worked = False
            for i, p_name in enumerate(param_set):
                if p_num == p_name:
                    data_and_nm[i][0].append(areas)
                    it_worked = True
            if not it_worked:
                print('something has gone horribly wrong')

    # Plot the areas data for all parameters on one plot to compare
    plot_areas(evols, data_and_nm, domain_name, graphs_fname)

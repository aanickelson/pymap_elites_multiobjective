import re

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon


def plot_it(x, y, iseff, fname, dirname):
    plt.clf()
    max_vals = [10, max(x), max(y)]
    # plt_max = max(max_vals) + 0.5
    plt_max = 21
    plt.xlim([-0.5, plt_max])
    plt.ylim([-0.5, plt_max])
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


def plot_areas(evos, areas_pareto, areas_no, areas_parallel, dirname, area_fname):
    print('We made it to plotting')
    plt.clf()
    evos = np.array(evos)
    set_for_loop = [[areas_pareto, 'pareto'], [areas_no, 'no'], [areas_parallel, 'parallel']]
    for [data, pareto] in set_for_loop:
        if not data:
            continue
        try:
            means, sterr = process(data)
            plt.plot(evos, means)
        except RuntimeWarning:
            print("'tis here, mlord")
            continue
        plt.fill_between(evos, means-sterr, means+sterr, alpha=0.5, label=pareto)
    plt.title(f"{dirname} areas")
    plt.legend()
    plt.savefig(os.path.join(area_fname, dirname + '.png'))


if __name__ == '__main__':
    # fname = '/home/anna/PycharmProjects/py_map_elites/examples/data/002_20230123_191216/archive_100000.dat'


    import os
    date = '20230201_135535'

    # rootdir = '/home/toothless/workspaces/pymap_elites_multiobjective/examples/data2'
    rootdir = f'/home/anna/PycharmProjects/pymap_elites_multiobjective/examples/{date}'
    graphs_fname = os.path.join(rootdir, 'graphs')
    area_fname = os.path.join(rootdir, 'area_graphs')
    evols = [i*10000 for i in range(31) if i > 0]
    x = 0
    pnum = 'rastrigin'
    # for pnum in ['006']:  # '004', '005',
    for subdir, dirs, files in os.walk(rootdir):
        print(subdir)
        for file in files:
            if 'centroids' in file:
                continue
            print(int(re.findall(r'\d+', file)[0]))

        if not dirs:
            continue
        # areas_pareto = []
        # areas_no = []
        # areas_parallel = []
        #
        # for sub in dirs:
        #     print(sub)
        #     pareto = 'no'
        #     if not pnum in sub:
        #         continue
        #     if 'pareto' in sub:
        #         pareto = 'pareto'
        #     elif 'parallel' in sub:
        #         pareto = 'parallel'
        #     areas = []
        #     for evo in evols2:
        #         fname = os.path.join(rootdir, sub, f'archive_{evo}.dat')
        #         # print(fname)
        #         try:
        #             x, y, xy = load_data(fname)
        #         except FileNotFoundError:
        #             continue
        #         is_eff = is_pareto_efficient_simple(xy)
        #         # curve_area = plot_it(x, y, is_eff, f'{sub}_{evo}', graphs_fname)
        #         curve_area = get_area(x[is_eff], y[is_eff])
        #         areas.append(curve_area)
        #     if len(areas) < 15:
        #         continue
        #     if pareto == 'pareto':
        #         areas_pareto.append(areas)
        #     elif pareto == 'no':
        #         areas_no.append(areas)
        #     elif pareto == 'parallel':
        #         areas_parallel.append(areas)
        #     else:
        #         print('something has gone horribly wrong')
        # plot_areas(evols, areas_pareto, areas_no, areas_parallel, pnum, area_fname)

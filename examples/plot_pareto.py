import numpy as np
from matplotlib import pyplot as plt


def plot_it(x, y, iseff, fname, dirname):
    plt.clf()
    max_vals = [10, max(x), max(y)]
    plt_max = max(max_vals) + 0.5
    plt.xlim([-0.5, plt_max])
    plt.ylim([-0.5, plt_max])
    plt.scatter(x, y, c='red')
    plt.scatter(x[iseff], y[iseff], c="blue")
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.title(fname)
    plt.savefig(os.path.join(dirname, fname + '.png'))


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


if __name__ == '__main__':
    # fname = '/home/anna/PycharmProjects/py_map_elites/examples/data/002_20230123_191216/archive_100000.dat'


    import os

    rootdir = '/home/anna/PycharmProjects/py_map_elites/examples/data'
    graphs_fname = os.path.join(rootdir, 'graphs')

    for subdir, dirs, files in os.walk(rootdir):
        for sub in dirs:
            fname = os.path.join(rootdir, sub, 'archive_100000.dat')
            try:
                x, y, xy = load_data(fname)
            except FileNotFoundError:
                continue
            is_eff = is_pareto_efficient_simple(xy)
            plot_it(x, y, is_eff, sub, graphs_fname)


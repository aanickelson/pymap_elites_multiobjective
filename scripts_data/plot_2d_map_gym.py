#! /usr/bin/env python
# | This file is a part of the pymap_elites framework.
# | Copyright 2019, INRIA
# | Main contributor(s):
# | Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
# | Eloise Dalin , eloise.dalin@inria.fr
# | Pierre Desreumaux , pierre.desreumaux@inria.fr
# |
# |
# | **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
# | mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
# |
# | This software is governed by the CeCILL license under French law
# | and abiding by the rules of distribution of free software.  You
# | can use, modify and/ or redistribute the software under the terms
# | of the CeCILL license as circulated by CEA, CNRS and INRIA at the
# | following URL "http://www.cecill.info".
# |
# | As a counterpart to the access to the source code and rights to
# | copy, modify and redistribute granted by the license, users are
# | provided only with a limited warranty and the software's author,
# | the holder of the economic rights, and the successive licensors
# | have only limited liability.
# |
# | In this respect, the user's attention is drawn to the risks
# | associated with loading, using, modifying and/or developing or
# | reproducing the software by the user in light of its specific
# | status of free software, that may mean that it is complicated to
# | manipulate, and that also therefore means that it is reserved for
# | developers and experienced professionals having in-depth computer
# | knowledge. Users are therefore encouraged to load and test the
# | software's suitability as regards their requirements in conditions
# | enabling the security of their systems and/or data to be ensured
# | and, more generally, to use and operate it in the same conditions
# | as regards security.
# |
# | The fact that you are presently reading this means that you have
# | had knowledge of the CeCILL license and that you accept its terms.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import KDTree
import pymap_elites_multiobjective.scripts_data.often_used as util
from pymap_elites_multiobjective.scripts_data.plot_pareto_gym import file_setup
import re
import os
from scipy import stats

# Type 1 / Truetype Fonts for GECCO
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Source: https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def load_data(filename, dim, n_fit):
    # print("\nLoading ", filename)
    data = np.loadtxt(filename)
    if len(data.shape) < 2:
        data = np.array([data])
    try:
        fit = data[:, 0:n_fit]
        desc = data[:, n_fit: dim + n_fit]
    except IndexError:
        return False

    return fit, desc


def load_centroids(filename):
    points = np.loadtxt(filename)
    return points


def plot_cvt(centroids, fit, desc, dim1, dim2, min_fit, max_fit, e, graph_f, sub_d, pctbh, reduced=False):
    fig, axes = plt.subplots(1, 1, figsize=(10, 10), facecolor='white', edgecolor='white')
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)

    # getting the original colormap using cm.get_cmap() function
    orig_map = plt.cm.get_cmap('viridis')

    # reversing the original colormap using reversed() function
    my_cmap = orig_map.reversed()

    # compute Voronoi tesselation
    # print("Voronoi...")
    vor = Voronoi(centroids[:, [dim1, dim2]])
    regions, vertices = voronoi_finite_polygons_2d(vor)
    # print("fit:", min_fit, max_fit)
    norm = mpl.colors.Normalize(vmin=min_fit, vmax=max_fit)
    # print("KD-Tree...")
    kdt = KDTree(centroids, leaf_size=30, metric='euclidean')

    print("plotting contour...")
    # ax.scatter(centroids[:, 0], centroids[:,1], c=fit)
    # contours
    for i, region in enumerate(regions):
        polygon = vertices[region]
        axes.fill(*zip(*polygon), alpha=0.05, edgecolor='black', facecolor='white', lw=1)

    print("plotting data...")
    k = 0
    cols = np.zeros(centroids.shape[0])
    for i in range(len(desc)):
        q = kdt.query([desc[i]], k=1)
        index = q[1][0][0]
        region = regions[index]
        polygon = vertices[region]
        if cols[index] < fit[i]:
            cols[index] = fit[i]
            axes.fill(*zip(*polygon), alpha=0.9, color=my_cmap(norm(cols[index])))
        k += 1
        if k % 100 == 0:
            print(k, end=" ", flush=True)
    fit_reshaped = fit.reshape((len(fit),))
    norm = mpl.colors.Normalize(vmin=0, vmax=max_fit)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap),
                 ax=axes, orientation='vertical', label='Distance to global Pareto front')
    sc = axes.scatter(desc[:, 0], desc[:, 1], c='b', s=1, zorder=0)
    print('\n')
    plt.title(f'Behavior Space, {pctbh:0.2f}')
    pre = 'bh_'
    if reduced:
        pre = 'red_bh_'
    for ex in e:
        figpath = os.path.join(graph_f, f'{pre}{sub_d}_dims{dim1}{dim2}_{ex}')
        fig.savefig(figpath)
    plt.clf()


def mk_files(rootdir, subd, niches, pols, n_bh):
    # Get the name of the sub-directory
    p_num = re.split('_|/', subd)[0]
    pth = os.path.join(rootdir, subd)

    cent_f = os.path.join(pth, f'centroids_{niches}_{n_bh}.dat')
    dat_f = os.path.join(pth, f'archive_{pols}.dat')

    if not os.path.exists(dat_f):
        print(f"File does not exist: {dat_f}")
        return False

    return [pth, cent_f, dat_f]


def calc_fit_data(fitnesses, layers, nobj):
    fit = np.zeros(fitnesses.shape[0]) + 0.000001
    fits = fitnesses.copy()
    x = 1
    for lay in range(layers, 0, -1):

        pareto = util.is_pareto_efficient_simple(fits)
        fit[pareto] = x
        fits[pareto] = [0] * nobj
        # Mountain
        x *= 0.965
        # Hopper
        # x *= 0.98
        # if not lay % 100:
        #     print(lay)
        if np.max(fits) < 0.000001:
            break
    return fit


def process_and_plot(files_info, ds, ext, sub, plotit, n_obj, n_pareto_layers, red=False):
    if not files_info:
        print(f'### SKIPPING')
        return
    [fpath, centroids_f, data_f, grph_f] = files_info
    # print(f'processing {fpath}')
    centroids = load_centroids(centroids_f)
    ftns, beh = load_data(data_f, centroids.shape[1], n_obj)
    _, counts = np.unique(beh, return_counts=True, axis=0)
    pct_bh = len(counts) / centroids.shape[0]
    # print(pct_bh)
    fit = calc_fit_data(ftns, n_pareto_layers, n_obj)

    if plotit:
        for dim01, dim02 in ds:
            # Plot
            plot_cvt(centroids, fit, beh, dim01, dim02, 0, 1, ext, grph_f, sub, pct_bh, red)

    return pct_bh


if __name__ == "__main__":

    exts = ['.svg','.png']  #

    # Hopper
    # gym_dir_name = 'hopper'
    # dates =  ['018_20240115_124423', '019_20240115_152734', '020_20240117_124709', '021_20240119_095648',
    #             '022_20240126_145552']
    # bh_dict = {'auto so ac': 2, 'auto mo ac': 2, 'auto so st': 2, 'auto mo st': 2,
    #            'avg act': 3, 'avg st': 4, 'fin act': 3, 'fin st': 4,
    #            'min max st': 8, 'min avg max st': 12, 'min max act': 6, 'min avg max act': 9}
    # param_nms = ['auto so ac', 'auto mo ac', 'auto so st', 'auto mo st',
    #              'avg st', 'fin st', 'avg act', 'fin act',
    #              'min max st', 'min avg max st', 'min max act', 'min avg max act']
    # param_nms = ['avg act', 'min max st']

    # Mountain
    gym_dir_name = 'mountain'
    dates = ['024_20240117_124709', '025_20240126_145552', '026_20240128_145330', '028_20240129_110622']
    bh_dict = {'auto so ac': 2, 'auto mo ac': 2, 'auto so st': 2, 'auto mo st': 2,
               'avg act':1, 'fin act':1, 'min max act': 2,'min avg max act':3,
               'avg st':2, 'fin st':2, 'min max st': 4, 'min avg max st': 6}
    param_nms = ['avg st', 'min max act']

    n_niches = 1000
    n_pols = 100000

    final_num = n_pols
    n_objectives = 2
    n_pareto_layers = 300
    plotornot = True


    # param_nms = ['avg act', 'avg st', 'fin act', 'fin st', 'min avg max act']
    # param_nms = ['avg st', 'fin st', 'min avg max act']
    # param_nms = ['min avg max act']
    dims = [[0, 1]]

    basedir_qd = os.path.join(os.getcwd(), 'data', gym_dir_name)
    graphs_f = file_setup(dates, cwd=basedir_qd)
    files_info = [[dates, basedir_qd, 'archive_']]

    param_sets = ['000']
    params_dict = {p: [] for p in param_nms}

    for date in dates:
        root_dir = os.path.join(basedir_qd, date)

        bh_pth = os.path.join(root_dir, 'bh_vis')
        # util.make_a_directory(bh_pth)
        # print(root_dir)
        sub_dirs = list(os.walk(root_dir))[0][1]
        for d in sub_dirs:
            p_num = re.split('_|/', d)[1]

            if p_num == 'auto so':
                p_num = 'auto so st'
            elif p_num == 'auto mo':
                pnum = 'auto mo st'
            if p_num not in params_dict:
                print(f'skipping {p_num} not in dict')
                continue
            bh_size = bh_dict[p_num]
            f_info = mk_files(root_dir, d, n_niches, n_pols, bh_size)
            if not f_info:
                continue
            f_info.append(graphs_f)
            try:
                d_date = date[:4] + d
                bh_fill = process_and_plot(f_info, dims, exts, d_date, plotornot, n_objectives, n_pareto_layers)
            except FileNotFoundError:
                continue
            params_dict[p_num].append(bh_fill)
            print(d, bh_fill)
    graphsfname = f_info[-1]
    with open(graphsfname + 'by type NOTES.TXT', 'w') as fl:
        for i, [key, val] in enumerate(params_dict.items()):
            fl.write(f'{key} & - & - & - & - & & &  {np.mean(val):0.5f} & {stats.sem(val):0.5f} \n')
    print(params_dict)

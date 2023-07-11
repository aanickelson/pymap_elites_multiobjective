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
import sys
from matplotlib.ticker import FuncFormatter
from sklearn.neighbors import KDTree
import matplotlib.cm as cm
import pymap_elites_multiobjective.scripts_data.often_used as util
from pymap_elites_multiobjective.scripts_data.plot_pareto import file_setup
import re



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


def load_data(filename, dim, dim_x, n_fit):
    print("\nLoading ", filename)
    data = np.loadtxt(filename)
    fit = data[:, 0:n_fit]
    desc = data[:, n_fit: dim + n_fit]
    x = data[:, dim + n_fit:dim + n_fit + dim_x]

    return fit, desc, x


def load_centroids(filename):
    points = np.loadtxt(filename)
    return points


def plot_cvt(ax, centroids, fit, desc, dim1, dim2, min_fit, max_fit):
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
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor='black', facecolor='white', lw=1)

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
            ax.fill(*zip(*polygon), alpha=0.9, color=my_cmap(norm(cols[index])))
        k += 1
        if k % 100 == 0:
            print(k, end=" ", flush=True)
    fit_reshaped = fit.reshape((len(fit),))
    norm = mpl.colors.Normalize(vmin=0, vmax=max_fit)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap),
                 ax=ax, orientation='vertical', label='Distance to global Pareto front')
    sc = ax.scatter(desc[:, 0], desc[:, 1], c='b', s=1, zorder=0)


def mk_files(rootdir, subd, niches, pols, bh_size=5):
    # Get the name of the sub-directory
    p_num = re.split('_|/', subd)[0]
    pth = os.path.join(rootdir, subd)

    cent_f = os.path.join(pth, f'centroids_{niches}_{bh_size}.dat')
    dat_f = os.path.join(pth, f'archive_{pols}.dat')

    if not os.path.exists(cent_f):
        bh_size = 6
        print("Trying behavior size 6")
        cent_f = os.path.join(pth, f'centroids_{niches}_{bh_size}.dat')
        if not os.path.exists(cent_f):
            return False
    if not os.path.exists(dat_f):
        print(f"File does not exist: {dat_f}")
        return False

    return pth, cent_f, dat_f


def calc_fit_data(fitnesses, layers):
    fit = np.ones(fitnesses.shape[0])
    fits = fitnesses.copy()
    for lay in range(layers, 0, -1):
        pareto = util.is_pareto_efficient_simple(fits)
        fit[pareto] = lay + 5
        fits[pareto] = [0, 0]
        if np.max(fitnesses) < 0.0001:
            break
    return fit


if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     sys.exit('Usage: %s centroids_file archive.dat [min_fit] [max_fit]' % sys.argv[0])
    import os
    dates = ['012_20230711_123339']
    ext = ['.png']  # ,'.png'
    n_niches = 5000
    n_pols = 200000
    final_num = 200000
    bh_size = 5
    n_objectives = 2
    n_pareto_layers = 50
    dim_x = 24
    param_sets = ['000', '009', '019', '119']

    params_dict = {pname: [] for pname in param_sets}

    graphs_f = os.path.join(file_setup(dates, 'bh_'), 'bh')
    for date in dates:
        root_dir = os.path.join(os.getcwd(), 'data', date)

        bh_pth = os.path.join(root_dir, 'bh_vis')
        if not os.path.exists(graphs_f):
            os.mkdir(graphs_f)
        util.make_a_directory(bh_pth)
        print(root_dir)
        sub_dirs = list(os.walk(root_dir))[0][1]
        for d in sub_dirs:
            p_num = re.split('_|/', d)[0]
            if p_num not in params_dict:
                continue

            files_info = mk_files(root_dir, d, n_niches, n_pols, bh_size)
            if not files_info:
                print(f'### SKIPPING {d}')
                continue

            print(f'processing {d}')
            fpath, centroids_f, data_f = files_info
            centroids = load_centroids(centroids_f)
            ftns, beh, x = load_data(data_f, centroids.shape[1], dim_x, n_objectives)
            _, counts = np.unique(beh, return_counts=True, axis=0)
            pct_bh = len(counts) / centroids.shape[0]
            fit = calc_fit_data(ftns, n_pareto_layers)

            if p_num in params_dict:
                params_dict[p_num].append(pct_bh)
            else:
                params_dict[p_num] = [pct_bh]

            # Plot
            fig, axes = plt.subplots(1, 1, figsize=(10, 10), facecolor='white', edgecolor='white')
            axes.set_xlim(0, 1)
            axes.set_ylim(0, 1)

            if p_num == '000':
                dim01 = 0
                dim02 = 3
            else:
                dim01 = 1
                dim02 = 3

            plot_cvt(axes, centroids, fit, beh, dim01, dim02, 0, (n_pareto_layers + 5))
            for ex in ext:
                plt.title(f'Behavior Space, {pct_bh*100:.02f}% filled')
                figpath = os.path.join(graphs_f, f'bh_{d}_dims{dim01}{dim02}_{date}_{ex}')
                fig.savefig(figpath)
                plt.clf()

    text_f = os.path.join(graphs_f, 'NOTES_bh.txt')
    with open(text_f, 'w') as f:
        for key, value in params_dict.items():
            f.write(f'{key}, {np.mean(value)}\n')

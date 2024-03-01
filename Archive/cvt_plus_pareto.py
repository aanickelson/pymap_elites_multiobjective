#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.

import math
import numpy as np
import multiprocessing

# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree

from pymap_elites_multiobjective.map_elites import common as cm


def __add_to_archive(s, centroid, archive, kdt):
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = cm.make_hashable(niche)
    s.centroid = n
    if n in archive:
        archive[n] = is_pareto_efficient_simple(archive[n], s)
        return
    else:
        archive[n] = [s]
        return


def __add_pareto_to_archive(s_list, archive, kdt):
    global_pareto = is_pareto_efficient_simple(s_list)
    for s in global_pareto:
        niche_index = kdt.query([s.desc], k=1)[1][0][0]
        niche = kdt.data[niche_index]
        n = cm.make_hashable(niche)
        s.centroid = n
        archive[n].append(s)
    return


def is_pareto_efficient_simple(vals, new=None):
    """
    copied and modified from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    if new:
        vals.append(new)
    fitnesses = [s.fitness for s in vals]
    costs = np.array(fitnesses)
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)  # Keep any point with a lower cost
            eff_add = np.all(costs == c, axis=1)
            is_efficient += eff_add
            is_efficient[i] = True  # And keep self
    final_vals = [vals[j] for j in range(len(vals)) if is_efficient[j]]
    return final_vals


# evaluate a single vector (x) with a function f and return a species
# t = vector, function
def __evaluate(t):
    z, f = t  # evaluate z with function f
    fit, desc = f(z)
    return cm.Species(z, desc, fit)

# map-elites algorithm (CVT variant)
def compute(dim_map, dim_x, f,
            n_niches=1000,
            max_evals=1e5,
            params=cm.default_params,
            log_file=None,
            variation_operator=cm.variation,
            data_fname=None):
    """CVT MAP-Elites
       Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

       Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile

    """
    # setup the parallel processing pool
    if params['parallel'] == True:

        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_cores)
    else:
        pool = None

    # create the CVT
    c = cm.cvt(n_niches, dim_map, params['cvt_samples'], data_fname, params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    cm.__write_centroids(c, data_fname)

    archive = {} # init archive (empty)
    n_evals = 0 # number of evaluations since the beginning
    b_evals = 0 # number evaluation since the last dump

    # main loop
    while n_evals < max_evals:
        to_evaluate = []
        # random initialization
        if len(archive) <= params['random_init'] * n_niches:
            for i in range(0, params['random_init_batch']):
                x = np.random.uniform(low=params['min'], high=params['max'], size=dim_x)
                to_evaluate += [(x, f)]
        else:  # variation/selection loop
            arch_pols = []
            # Unpack all the policies to mutate
            for key, value in archive.items():
                for val_i, val in enumerate(value):
                    arch_pols.append(val)

            # we select all the parents at the same time because randint is slow
            rand1 = np.random.randint(len(arch_pols), size=params['batch_size'])
            rand2 = np.random.randint(len(arch_pols), size=params['batch_size'])
            for n in range(0, params['batch_size']):
                # parent selection
                x = arch_pols[rand1[n]]
                y = arch_pols[rand2[n]]
                # copy & add variation
                z = variation_operator(x.x, y.x, params)
                to_evaluate += [(z, f)]

        # evaluation of the fitness for to_evaluate
        s_list = cm.parallel_eval(__evaluate, to_evaluate, pool, params)
        # natural selection
        for s in s_list:
            __add_to_archive(s, s.desc, archive, kdt)

        __add_pareto_to_archive(s_list, archive, kdt)
        # count evals
        n_evals += len(to_evaluate)
        b_evals += len(to_evaluate)

        # write archive
        if b_evals >= params['dump_period'] and params['dump_period'] != -1:
            print(f"[{n_evals}/{int(max_evals)}] - {data_fname} \n", end=" ", flush=True)
            cm.__save_archive(archive, n_evals, data_fname)
            b_evals = 0
        # write log
        if log_file != None:
            fit_list = np.array([x[0].fitness[0] for x in archive.values()])
            log_file.write("{} {} {} {} {} {} {}\n".format(n_evals, len(archive.keys()),
                    fit_list.max(), np.mean(fit_list), np.median(fit_list),
                    np.percentile(fit_list, 5), np.percentile(fit_list, 95)))
            log_file.flush()
    cm.__save_archive(archive, n_evals, data_fname, final=True)
    return archive

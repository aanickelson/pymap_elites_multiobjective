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
from datetime import datetime
import random
# from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree

from pymap_elites_multiobjective.map_elites import common as cm
from evo_playground.support.auto_encoder import Autoencoder


def __add_to_archive(slist, archive, kdt, pareto=True):
    niches_changed = []
    for s in slist:
        centroid = s.desc
        try:
            niche_index = kdt.query([centroid], k=1)[1][0][0]
        except ValueError:
            print("Value Error at kdt query for centroid (cvt > add to archive)")
            print(centroid)
            print(s.desc, s.centroid, s.fitness)
            exit(1)
        niche_kdt = kdt.data[niche_index]
        n = cm.make_hashable(niche_kdt)
        s.centroid = n
        if n in archive:
            archive[n].append(s)
            # archive[n] = is_pareto_efficient_simple(archive[n], s)
        else:
            archive[n] = [s]

        if n not in niches_changed:
            niches_changed.append(n)

    if pareto:
        for ni in niches_changed:
            archive[ni] = is_pareto_efficient_simple(archive[ni])
    else:
        for ni in niches_changed:
            max_pol = np.argmax([sum(pol.fitness) for pol in archive[ni]])
            archive[ni] = [archive[ni][max_pol]]


def is_pareto_efficient_simple(vals):
    """
    copied and modified from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    n_to_keep = 50
    fitnesses = [s.fitness for s in vals]
    costs = np.array(fitnesses)
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    err = 0.0005
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)  # Keep any point with a higher cost
            # The two lines below this capture points that are equal to the current compared point
            # Without these two lines, it will only keep one of each pareto point
            # E.g. if there are two policies that both get [4,0], only the first will be kept. That's bad.
            if np.all(c < err):
                is_efficient[i] = True
                continue
            eff_add = np.all(abs(costs - c) < err, axis=1)
            is_efficient += eff_add
            is_efficient[i] = True  # And keep self
    pareto_vals = [vals[j] for j in range(len(vals)) if is_efficient[j]]
    pareto_costs = np.array([costs[k] for k in range(len(vals)) if is_efficient[k]])

    # If the list is too big.
    if len(pareto_vals) > n_to_keep:
        # Find indices of unique pareto values
        _, idxs = np.unique(pareto_costs, axis=0, return_index=True)
        # Keep those
        final_vals = [pareto_vals[l] for l in idxs]
        # If there's still room left, randomly sample from the entire list
        if len(final_vals) < n_to_keep:
            sample_set = random.sample(pareto_vals, n_to_keep - len(final_vals))
            final_vals.extend(sample_set)
        # If the list is too big, randomly sample to keep the right number
        elif len(final_vals) > n_to_keep:
            final_vals = random.sample(final_vals, n_to_keep)
        # If it's exactly equal, you're good.
        else:
            pass

    # The list is not too big, keep everything
    else:
        final_vals = pareto_vals

    return final_vals

# evaluate a single vector (x) with a function f and return a species
# t = vector, function
def __evaluate(t):
    """
    Change to this version:
    Instead of filling in the description (bh) at this stage, it only fills in the states.
    This will be used by the auto-encoder to evaluate the description (bh)
    """
    z, f = t  # evaluate z with function f
    fit, desc = f(z)
    return cm.Species(z, desc=[], fitness=fit, states=desc)

def get_bh_from_auto(autoenc, archive, new_pols):
    # List of species in archive
    arch_list = []
    for v in list(archive.values()):
        arch_list.extend(v)
    # arch_list = list(archive.values())
    arch_list.extend(new_pols)              # Add the newly tested policies
    st_list = np.array([a.states for a in arch_list])

    # Train the auto-encoder
    autoenc.train(st_list)


    model_out = autoenc.feed(st_list)       # get the behavior descriptors from the autoencoder
    for i, sp in enumerate(arch_list):                 # Set the behavior descriptors for each species
        sp.desc = model_out[i]
    archive = {}                            # Reset the archive
    return archive, autoenc, arch_list


# map-elites algorithm (CVT variant)
def compute(bh_size, dim_x, wrapper,
            n_niches=1000,
            max_evals=1e5,
            params=cm.default_params,
            log_file=None,
            variation_operator=cm.variation,
            data_fname=None,
            multiobj=True):
    """CVT MAP-Elites
       Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

       Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile

    """
    # Set the function that will be called
    f = wrapper.run_bh

    # Set up the autoencoder
    autoencoder = Autoencoder(wrapper.bh_size(wrapper.bh_name))

    # setup the parallel processing pool
    if params['parallel']:
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_cores)
    else:
        pool = None

    # create the CVT
    c = cm.cvt(n_niches, bh_size,
               params['cvt_samples'], data_fname, params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    cm.__write_centroids(c, data_fname)

    archive = {}    # init archive (empty)
    n_evals = 0     # number of evaluations since the beginning
    b_evals = 0     # number evaluation since the last dump

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

        # Train the autoencoder
        # Then empty the archive and get the new bh descriptors from the autoencoder
        archive, autoencoder, s_list = get_bh_from_auto(autoencoder, archive, s_list)

        # natural selection
        __add_to_archive(s_list, archive, kdt, multiobj)
        # count evals
        n_evals += len(to_evaluate)
        b_evals += len(to_evaluate)

        # write archive
        if n_evals < 1000:
            now = datetime.now()
            now_str = now.strftime("%H:%M:%S")
            print(f"[{n_evals}/{int(max_evals)}] - {now_str} - {data_fname} \n", end=" ", flush=True)
        if b_evals >= params['dump_period'] != -1:
            now = datetime.now()
            now_str = now.strftime("%H:%M:%S")
            print(f"[{n_evals}/{int(max_evals)}] - {now_str} - {data_fname} \n", end=" ", flush=True)
            cm.__save_archive(archive, n_evals, data_fname)
            b_evals = 0
        # write log
        if log_file is not None:
            fit_list = np.array([x[0].fitness[0] for x in archive.values()])
            log_file.write("{} {} {} {} {} {} {}\n".format(n_evals, len(archive.keys()),
                                                           fit_list.max(), np.mean(fit_list), np.median(fit_list),
                                                           np.percentile(fit_list, 5), np.percentile(fit_list, 95)))
            log_file.flush()
    cm.__save_archive(archive, n_evals, data_fname, final=True)
    return archive

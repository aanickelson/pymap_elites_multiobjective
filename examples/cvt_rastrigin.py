# only required to run python3 examples/cvt_rastrigin.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import math

import pymap_elites_multiobjective.map_elites.cvt_plus_pareto as cvt_me_pareto
import pymap_elites_multiobjective.map_elites.cvt as cvt_me
import pymap_elites_multiobjective.map_elites.cvt_pareto_parallel as cvt_me_pareto_parallel
import pymap_elites_multiobjective.map_elites.common as cm_map_elites
from datetime import datetime
from os import path, getcwd, mkdir
import multiprocessing


def rastrigin(xx):
    x = xx * 10 - 5 # scaling to [-5, 5]
    f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
    return -f, np.array([xx[0], xx[1]])


def rastrigin2d(xx):
    x = np.zeros_like(xx)
    for i, x_i in enumerate(xx):
        if x_i > 4.0:
            x_i = 4.0
        elif x_i < -2.0:
            x_i = -2.0
        x[i] = x_i

    lbd1 = 0.0
    lbd2 = 2.2
    f1 = ((x - lbd1) ** 2 - 10 * np.cos(2 * math.pi * (x - lbd1))).sum()
    f2 = ((x - lbd2) ** 2 - 10 * np.cos(2 * math.pi * (x - lbd2))).sum()
    return [-f1, -f2], np.array([xx[0], xx[1]])


def main(setup):
    [cvt_p, filepath, with_pareto] = setup

    n_niches = 15000
    niche_desc_size = 2
    inputs_size = 10

    if with_pareto == 'pareto':
        print(with_pareto, filepath)
        archive = cvt_me_pareto.compute(niche_desc_size, inputs_size, rastrigin2d, n_niches=n_niches, max_evals=evals,
                                        log_file=open('cvt.dat', 'w'), params=cvt_p, data_fname=filepath)
    elif with_pareto == 'parallel':
        print(with_pareto, filepath)
        archive = cvt_me_pareto_parallel.compute(niche_desc_size, inputs_size, rastrigin2d, n_niches=n_niches, max_evals=evals,
                                                 log_file=open('cvt.dat', 'w'), params=cvt_p, data_fname=filepath)
    elif with_pareto == 'no':
        print(with_pareto, filepath)
        archive = cvt_me.compute(niche_desc_size, inputs_size, rastrigin2d, n_niches=n_niches, max_evals=evals,
                                 log_file=open('cvt.dat', 'w'), params=cvt_p, data_fname=filepath)
    else:
        print(f'{with_pareto} is not an option. Options are "parallel", "pareto", and "no"')


def multiprocess_main(batch_for_multi):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
        pool.map(main, batch_for_multi)


if __name__ == '__main__':

    # we do 10M evaluations, which takes a while in Python (but it is very fast in the C++ version...)
    px = cm_map_elites.default_params.copy()
    px["dump_period"] = 10000
    px["batch_size"] = 100
    px["min"] = -5
    px["max"] = 5
    px["parallel"] = False
    px['cvt_use_cache'] = False
    px['add_random'] = 5
    px['random_init_batch'] = 100
    px['random_init'] = 0.01    # Percent of niches that should be filled in order to start mutation
    evals = 300000

    batch = []
    pareto_paralell_options = ['parallel', 'no']  # 'pareto',
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    dirpath = path.join(getcwd(), now_str)
    mkdir(dirpath)

    for with_pareto in pareto_paralell_options:
        for i in range(50):
            filepath = path.join(dirpath, f'rastrigin2d_{with_pareto}_run{i}')
            mkdir(filepath)
            batch.append([px, filepath, with_pareto])

    # Use this one
    multiprocess_main(batch)

    # for p_set in batch:
    #     main(p_set)
# only required to run python3 Archive/cvt_rastrigin.py
import random
import sys, os

import numpy.random

import numpy as np
import math
from copy import deepcopy
from time import time
from itertools import combinations

from AIC.aic import aic as Domain
import pymap_elites_multiobjective.parameters as Params

import pymap_elites_multiobjective.map_elites.cvt as cvt_me
import pymap_elites_multiobjective.map_elites.common as cm_map_elites
from pymap_elites_multiobjective.scripts_data.run_env import run_env
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp
from evo_playground.support.rover_wrapper import RoverWrapper
from evo_playground.support.neuralnet import NeuralNetwork as NN
from torch import from_numpy
from datetime import datetime
from os import path, getcwd, mkdir
import multiprocessing
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main(setup):
    [env_p, cvt_p, filepath, stat_num, bh_strs] = setup
    print(f"main has begun for {stat_num}")
    numpy.random.seed(stat_num + random.randint(0, 10000))
    archive = {}
    bh_sizes = {'battery': 1, 'distance': 1, 'type sep': 4, 'type combo': 2,
                'v or e': 2, 'full act': 10}
    env_p.n_bh = 0
    for bhstr in bh_strs:
        env_p.n_bh += bh_sizes[bhstr]

    env = Domain(env_p)
    dom = RoverWrapper(env, bh_strs)

    # Dimension of x to be tested is the sum of the sizes of the weights vectors and bias vectors
    wts_dim = dom.agents[0].w0_size + dom.agents[0].w2_size + dom.agents[0].b0_size + dom.agents[0].b2_size
    n_niches = px['n_niches']

    start = time()
    archive = cvt_me.compute(env_p.n_bh, wts_dim, dom._evaluate_multiple, n_niches=n_niches, max_evals=evals,
                             log_file=open('cvt.dat', 'w'), params=cvt_p, data_fname=filepath)
    # tot_time = time() - start
    # with open(filepath + '_time.txt', 'w') as f:
    #     f.write(str(tot_time))


def multiprocess_main(batch_for_multi):
    cpus = multiprocessing.cpu_count() - 1
    # cpus = 4
    with multiprocessing.Pool(processes=cpus) as pool:
        pool.map(main, batch_for_multi)


def get_unique_fname(rootdir, date_time=None):
    greatest = 0
    # Walk through all the files in the given directory
    for sub, dirs, files in os.walk(rootdir):
        for d in dirs:
            pos = 3
            splitstr = re.split('_|/', d)
            try:
                int_str = int(splitstr[-pos])
            except (ValueError, IndexError):
                continue

            if int_str > greatest:
                greatest = int_str
        break

    return os.path.join(rootdir, f'{greatest + 1:03d}{date_time}')


if __name__ == '__main__':
    x = multiprocessing.cpu_count()
    # we do 10M evaluations, which takes a while in Python (but it is very fast in the C++ version...)
    px = cm_map_elites.default_params.copy()
    px["min"] = -5
    px["max"] = 5
    px["parallel"] = False
    px['cvt_use_cache'] = False
    px['add_random'] = 0
    px['random_init_batch'] = 100
    px['random_init'] = 0.001  # Percent of niches that should be filled in order to start mutation

    # RUN VALS:
    px["batch_size"] = 100
    px["dump_period"] = 10000
    px['n_niches'] = 1000
    evals = 100000

    # DEBUGGING VALS:
    # px["batch_size"] = 100
    # px["dump_period"] = 1000
    # px['n_niches'] = 100
    # evals = 10


    bh_options = ['battery', 'distance', 'type sep', 'type combo', 'v or e', 'full act']
    bh_combos = list(combinations(bh_options, 2))
    bh_options_one = [['type sep'], ['type combo'], ['v or e'], ['full act']]

    all_options = bh_options_one + bh_combos
    for niches in [1000, 2000]:
        print(f'Testing {niches}')
        px['n_niches'] = niches
        now = datetime.now()
        base_path = path.join(getcwd(), 'data')
        if not os.path.exists(base_path):
            mkdir(base_path)

        now_str = now.strftime("_%Y%m%d_%H%M%S")
        dirpath = get_unique_fname(base_path, now_str)
        # dirpath = path.join(getcwd(), now_str)
        mkdir(dirpath)
        batch = []

        for params in [Params.p200100, Params.p200000, Params.p211101]:  # Params.p200100,  Params.p111107:  # , Params.p119, Params.p211, Params.p219]:
            p = deepcopy(params)
            p.n_cf_evals = 1
            # p.cf_bh = False
            # p.n_bh = 2
            # if p.cf_bh:
            #     p.n_bh = params.n_poi_types + 3
            # else:
            #     p.n_bh = params.n_poi_types * 3
            p.n_agents = 1
            p.battery = 18
            lp.n_stat_runs = 15
            if p.counter == 0:
                p.n_cf_evals = 1
            # else:
            #     p.n_cf_evals = 10
            for c in all_options:
                combo_str = ''
                for c0 in c:
                    combo_str += c0 + '_'
                for i in range(lp.n_stat_runs):
                    filepath = path.join(dirpath, f'{p.param_idx:03d}_{combo_str}run{i}')
                    mkdir(filepath)
                    batch.append([p, px, filepath, i, c])

        # Use this one
        multiprocess_main(batch)

    # This runs a single experiment / setup at a time for debugging
    # px["parallel"] = False
    # main(batch[0])

    # px["parallel"] = True
    # for b in batch:
    #     main(b)

    # This is the bad way. Don't do it this way
    # num_cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(num_cores)
    # with multiprocessing.Pool(num_cores=multiprocessing.cpu_count()-1) as pool:
    #     pool.map(main, batch)

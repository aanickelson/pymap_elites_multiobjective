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
import pymap_elites_multiobjective.map_elites.cvt_auto_encoder as cvt_auto_encoder
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp
from evo_playground.support.rover_wrapper import RoverWrapper
from pymap_elites_multiobjective.scripts_data.sar_wrapper import SARWrap
from pymap_elites_multiobjective.cvt_params.mome_default_params import default_params
from datetime import datetime
from os import path, getcwd, mkdir
import multiprocessing
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main(setup):
    [env_p, cvt_p, filepath, stat_num, bh_name] = setup
    print(f"main has begun for {stat_num}")
    numpy.random.seed(stat_num + random.randint(0, 10000))
    archive = {}
    # bh_sizes = {'battery': 1, 'distance': 1, 'type sep': 4, 'type combo': 2,
    #             'v or e': 2, 'full act': 10}

    env = Domain(env_p)
    wrap = RoverWrapper(env, bh_name)

    # Dimension of x to be tested is the sum of the sizes of the weights vectors and bias vectors
    wts_dim = wrap.agents[0].w0_size + wrap.agents[0].w2_size + wrap.agents[0].b0_size + wrap.agents[0].b2_size
    n_niches = px['n_niches']
    if 'auto' in bh_name:
        n_behaviors = 2
        multiobjective = 'auto mo' in bh_name
        archive = cvt_auto_encoder.compute(n_behaviors, wts_dim, wrap, n_niches=px['n_niches'], max_evals=cvt_p["evals"],
                                 log_file=open('cvt.dat', 'w'), params=cvt_p, data_fname=filepath, multiobj=multiobjective)

    else:
        n_behaviors = wrap.bh_size(wrap.bh_name)
        archive = cvt_me.compute(n_behaviors, wts_dim, wrap.run_bh, n_niches=px['n_niches'], max_evals=cvt_p["evals"],
                                 log_file=open('cvt.dat', 'w'), params=cvt_p, data_fname=filepath)


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
    px = default_params.copy()

    # DEBUGGING VALS:
    # px["batch_size"] = 100
    # px["dump_period"] = 1000
    # px['n_niches'] = 100
    # evals = 10
    evals = 50000


    bh_options = ['auto so st', 'auto mo st','auto so ac', 'auto mo ac',
                  'avg st', 'fin st', 'min max st', 'min avg max st',
                  'avg act', 'fin act', 'min max act', 'min avg max act']
    # bh_options = ['battery', 'distance', 'type sep', 'type combo', 'v or e', 'full act']
    # bh_combos = list(combinations(bh_options, 2))
    # bh_options_one = [['type sep'], ['type combo'], ['v or e'], ['full act']]
    # all_options = bh_options_one + bh_combos
    all_options = bh_options

    for niches in [px['n_niches']]:
        print(f'Testing {niches}')
        px['n_niches'] = niches
        now = datetime.now()
        base_path = path.join(getcwd(), 'data', 'rover')
        if not os.path.exists(base_path):
            mkdir(base_path)

        now_str = now.strftime("_%Y%m%d_%H%M%S")
        dirpath = get_unique_fname(base_path, now_str)
        # dirpath = path.join(getcwd(), now_str)
        mkdir(dirpath)
        batch = []

        for params in [Params.p200000]:
            p = deepcopy(params)
            p.n_cf_evals = 1
            p.n_agents = 1
            p.battery = 18      # Found through experimentation
            lp.n_stat_runs = 1
            for bh in all_options:
                for i in range(lp.n_stat_runs):
                    filepath = path.join(dirpath, f'{p.param_idx:03d}_{bh}_run{i}')
                    mkdir(filepath)
                    batch.append([p, px, filepath, i, bh])

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

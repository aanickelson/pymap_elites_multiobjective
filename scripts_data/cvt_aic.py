# only required to run python3 Archive/cvt_rastrigin.py
import random
import sys, os

import numpy.random

from copy import deepcopy

from AIC.aic import aic as Domain
import pymap_elites_multiobjective.parameters as Params

import pymap_elites_multiobjective.map_elites.cvt as cvt_me
import pymap_elites_multiobjective.map_elites.cvt_auto_encoder as cvt_auto_encoder
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp
from pymap_elites_multiobjective.scripts_data.rover_wrapper import RoverWrapper
from pymap_elites_multiobjective.cvt_params.mome_default_params import default_params
import pymap_elites_multiobjective.scripts_data.often_used as oft

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
    px = default_params.copy()

    data_fpath = path.join(getcwd(), 'data')
    oft.make_a_directory(data_fpath)  # This makes the directory only if it does not already exist

    # DEBUGGING VALS:
    # px["batch_size"] = 100
    # px["dump_period"] = 1000
    # px['n_niches'] = 100
    # px["evals"] = 1000
    px["evals"] = 100000
    px['auto_batch'] = 10000

    # Select which behaviors you want to test
    # bh_options = ['avg act', 'fin act', 'min max act', 'min avg max act', 'auto so ac', 'auto mo ac',]
    # bh_options = ['avg st', 'fin st', 'min max st', 'min avg max st',
    #               'avg act', 'fin act', 'min max act', 'min avg max act',
    #                'auto mo st','auto mo ac',]  #, 'auto so st', 'auto so ac',
    bh_options = ['auto mo ac', 'auto so st', 'auto so ac']

    # Number of stat runs per behavior
    lp.n_stat_runs = 10

    ######################################################
    # You shouldn't need to change anything below here
    # Unless you want to run only one experiment for debugging purposes (go to the end comment block)
    ######################################################

    batch = []
    params = Params.p200000
    p = deepcopy(params)

    # Set up directories for data
    now = datetime.now()
    base_path = path.join(data_fpath, 'rover')
    oft.make_a_directory(base_path)  # This makes the directory only if it does not already exist

    now_str = now.strftime("_%Y%m%d_%H%M%S")
    dirpath = get_unique_fname(base_path, now_str)
    oft.make_a_directory(dirpath)

    # Set up the batch parameters to run
    for bh in bh_options:
        for i in range(lp.n_stat_runs):
            filepath = path.join(dirpath, f'{p.param_idx}_{bh}_run{i:02d}')
            oft.make_a_directory(filepath)  # This makes the directory only if it does not already exist
            batch.append([p, px, filepath, i, bh])

    ######################################################
    # Comment / uncomment the sections below depending on if you want to use multiprocessing
    ######################################################

    # Use this one to multiprocess all the experiments
    multiprocess_main(batch)

    # This runs a single experiment for debugging
    # 'parallel' flag determines if you test the environments in parallel, i.e. the policy evaluation
    # cannot be combined with multiprocessing
    # px["parallel"] = False
    # main(batch[0])

    # This runs one at a time for debugging
    # px["parallel"] = True
    # for b in batch:
    #     main(b)

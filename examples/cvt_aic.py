# only required to run python3 examples/cvt_rastrigin.py
import random
import sys, os

import numpy.random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import math
from copy import deepcopy

import pymap_elites_multiobjective.map_elites.cvt_plus_pareto as cvt_me_pareto
import pymap_elites_multiobjective.map_elites.cvt as cvt_me
import pymap_elites_multiobjective.map_elites.common as cm_map_elites
import pymap_elites_multiobjective.map_elites.cvt_pareto_parallel as cvt_me_pareto_parallel

from AIC.aic import aic as Domain
from pymap_elites_multiobjective.parameters import batch_00, batch_01

from pymap_elites_multiobjective.examples.run_env import run_env
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp
# from evo_playground.learning.neuralnet_no_hid import NeuralNetwork as NN
from evo_playground.learning.neuralnet import NeuralNetwork as NN
from torch import from_numpy
from datetime import datetime
from os import path, getcwd, mkdir
import multiprocessing


class RoverWrapper:
    def __init__(self, env, param):
        self.env = env
        self.p = param
        self.st_size = env.state_size()
        self.hid = lp.hid
        self.act_size = env.action_size()
        self.l1_size = self.st_size * self.hid
        self.l2_size = self.hid * self.act_size
        self.model = NN(self.st_size, self.hid, self.act_size)
        self.n_evals = 0

    def evaluate(self, x):
        self.env.reset()
        l1_wts = from_numpy(np.reshape(x[:self.l1_size], (self.hid, self.st_size)))
        l2_wts = from_numpy(np.reshape(x[self.l1_size:], (self.act_size, self.hid)))
        self.model.set_weights([l1_wts, l2_wts])
        fitness, bh = run_env(self.env, [self.model], self.p, use_bh=True)
        # fitness = self.env.multiG()
        # rmtime = self.env.agent_room_times()[0]

        self.n_evals += 1
        # if not self.n_evals % 1000:
        #     print(f"Number of values tested: {self.n_evals}")
        return fitness, bh[0]


def main(setup):
    [env_p, cvt_p, filepath, with_pareto, stat_num] = setup
    numpy.random.seed(stat_num + random.randint(0, 10000))
    archive = {}
    env = Domain(env_p)
    in_size = env.state_size()
    hid_size = lp.hid
    out_size = env.action_size()
    wts_dim = (in_size * hid_size) + (hid_size * out_size)
    dom = RoverWrapper(env, env_p)
    n_niches = px['n_niches']

    n_behaviors = env_p.n_bh
    archive = cvt_me.compute(n_behaviors, wts_dim, dom.evaluate, n_niches=n_niches, max_evals=evals,
                             log_file=open('cvt.dat', 'w'), params=cvt_p, data_fname=filepath)


def multiprocess_main(batch_for_multi):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
        pool.map(main, batch_for_multi)


if __name__ == '__main__':

    # we do 10M evaluations, which takes a while in Python (but it is very fast in the C++ version...)
    px = cm_map_elites.default_params.copy()
    px["min"] = -5
    px["max"] = 5
    px["parallel"] = False
    px['cvt_use_cache'] = False
    px['add_random'] = 0
    px['random_init_batch'] = 100
    px['random_init'] = 0.001    # Percent of niches that should be filled in order to start mutation

    # RUN VALS:
    px["batch_size"] = 100
    px["dump_period"] = 10000
    px['n_niches'] = 10000
    evals = 200000

    # DEBUGGING VALS:
    # px["batch_size"] = 10
    # px["dump_period"] = 100
    # px['n_niches'] = 100
    # evals = 200

    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    dirpath = path.join(getcwd(), now_str)
    mkdir(dirpath)
    # run one batch, then the other
    for param_batch in [batch_01, batch_00]:
        batch = []

        for params in param_batch:  #, p04]:
            p = deepcopy(params)
            if params.counter:
                p.n_bh = params.n_poi_types + 3
            else:
                p.n_bh = params.n_poi_types * 3
            p.n_agents = 1
            lp.n_stat_runs = 5
            pareto_paralell_options = ['no']  # 'no', 'pareto',, 'parallel',

            for with_pareto in pareto_paralell_options:
                for i in range(lp.n_stat_runs):
                    filepath = path.join(dirpath, f'{p.param_idx:03d}_{with_pareto}_run{i}')
                    mkdir(filepath)
                    batch.append([p, px, filepath, with_pareto, i])

        # Use this one
        multiprocess_main(batch)

    # This runs a single experiment / setup at a time for debugging
    # main(batch[0])

    # for b in batch:
    #     main(b)


    # This is the bad way. Don't do it this way
    # num_cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(num_cores)
    # with multiprocessing.Pool(num_cores=multiprocessing.cpu_count()-1) as pool:
    #     pool.map(main, batch)


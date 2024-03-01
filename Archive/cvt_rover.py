# only required to run python3 Archive/cvt_rastrigin.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import pymap_elites_multiobjective.Archive.cvt_plus_pareto as cvt_me_pareto
import pymap_elites_multiobjective.map_elites.cvt as cvt_me
import pymap_elites_multiobjective.map_elites.common as cm_map_elites
import pymap_elites_multiobjective.Archive.cvt_pareto_parallel as cvt_me_pareto_parallel

from teaming.domain import DiscreteRoverDomain as Domain
import evo_playground.parameters as param
from evo_playground.support.neuralnet_no_hid import NeuralNetwork as NN
from torch import from_numpy
from datetime import datetime
from os import path, getcwd, mkdir
import multiprocessing


class RoverWrapper:
    def __init__(self, env):
        self.env = env
        self.st_size = env.state_size()
        self.hid = env.p.hid
        self.act_size = env.get_action_size()
        self.l1_size = self.st_size * self.act_size
        self.l2_size = self.hid * self.act_size
        self.model = NN(self.st_size, self.hid, self.act_size)
        self.n_evals = 0

    def evaluate(self, x):
        self.env.reset()
        l1_wts = from_numpy(np.reshape(x[:self.l1_size], (self.act_size, self.st_size)))
        # l2_wts = from_numpy(np.reshape(x[self.l1_size:], (self.act_size, self.hid)))
        self.model.set_weights([l1_wts])
        self.env.run_sim([self.model])
        fitness = self.env.multiG()
        rmtime = self.env.agent_room_times()[0]
        self.n_evals += 1
        # if not self.n_evals % 1000:
        #     print(f"Number of values tested: {self.n_evals}")
        return fitness, rmtime


def main(setup):
    [env_p, cvt_p, n_niches, filepath, with_pareto] = setup
    archive = {}
    env = Domain(env_p)
    in_size = env.state_size()
    out_size = env.get_action_size()
    wts_dim = in_size * out_size
    dom = RoverWrapper(env)
    if with_pareto == 'pareto':
        print(with_pareto, filepath)
        archive = cvt_me_pareto.compute(env.n_rooms, wts_dim, dom.evaluate, n_niches=n_niches, max_evals=evals,
                                        log_file=open('cvt.dat', 'w'), params=cvt_p, data_fname=filepath)
    elif with_pareto == 'parallel':
        print(with_pareto, filepath)
        archive = cvt_me_pareto_parallel.compute(env.n_rooms, wts_dim, dom.evaluate, n_niches=n_niches, max_evals=evals,
                                                 log_file=open('cvt.dat', 'w'), params=cvt_p, data_fname=filepath)
    elif with_pareto == 'no':
        print(with_pareto, filepath)
        archive = cvt_me.compute(env.n_rooms, wts_dim, dom.evaluate, n_niches=n_niches, max_evals=evals,
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
    px['random_init'] = 0.001    # Percent of niches that should be filled in order to start mutation
    evals = 500000

    batch = []
    pareto_paralell_options = ['parallel', 'no']  # 'pareto',
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    dirpath = path.join(getcwd(), now_str)
    mkdir(dirpath)
    n_niches = [100, 1000, 10000]
    for with_pareto in pareto_paralell_options:
        for nich in n_niches:
            for p in [param.p06]:  # param.p04, param.p05,
                for i in range(3):
                    filepath = path.join(dirpath, f'{p.trial_num:03d}_{with_pareto}_niches{nich}_run{i}')
                    mkdir(filepath)
                    batch.append([p, px, nich, filepath, with_pareto])

    # Use this one
    multiprocess_main(batch)

    # This runs a single experiment / setup at a time for debugging
    # for setup in batch:
    #     main(setup)


    # This is the bad way. Don't do it this way
    # num_cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(num_cores)
    # with multiprocessing.Pool(num_cores=multiprocessing.cpu_count()-1) as pool:
    #     pool.map(main, batch)


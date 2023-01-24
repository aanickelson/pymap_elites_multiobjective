# only required to run python3 examples/cvt_rastrigin.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import math

import py_map_elites.map_elites.cvt as cvt_map_elites
import py_map_elites.map_elites.common as cm_map_elites
from teaming.domain import DiscreteRoverDomain as Domain
from evo_playground.parameters.parameters04 import Parameters as p4
from evo_playground.parameters.parameters02 import Parameters as p2
from evo_playground.learning.neuralnet_no_hid import NeuralNetwork as NN
from torch import from_numpy
from datetime import datetime
from os import path, getcwd, mkdir


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
        if not self.n_evals % 1000:
            print(f"Number of values tested: {self.n_evals}")
        return fitness, rmtime


if __name__ == '__main__':

    # we do 10M evaluations, which takes a while in Python (but it is very fast in the C++ version...)
    px = cm_map_elites.default_params.copy()
    px["dump_period"] = 5000
    px["batch_size"] = 2
    px["min"] = 0
    px["max"] = 1
    px["parallel"] = True
    px['cvt_use_cache'] = False
    evals = 100000
    for _ in range(5):
        for p in [p2, p4]:

            env = Domain(p)
            in_size = env.state_size()
            out_size = env.get_action_size()
            wts_dim = in_size * out_size
            dom = RoverWrapper(env)
            now = datetime.now()
            now_str = now.strftime("%Y%m%d_%H%M%S")
            filepath = path.join(getcwd(), 'data', f'{p.trial_num:03d}_{now_str}')
            mkdir(filepath)

            archive = cvt_map_elites.compute(env.n_rooms, wts_dim, dom.evaluate, n_niches=500, max_evals=evals, log_file=open('cvt.dat', 'w'), params=px, data_fname=filepath)

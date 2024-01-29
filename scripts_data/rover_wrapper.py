import os
import sys

import numpy as np
from torch import from_numpy

from evo_playground.support.neuralnet import NeuralNetwork as NN
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp



class AgPol:
    def __init__(self, st_size, hid, act_size):
        self.st_size = st_size
        self.hid = hid
        self.act_size = act_size
        self.w0_size = self.st_size * self.hid
        self.w2_size = self.hid * self.act_size
        self.b0_size = self.hid
        self.b2_size = self.act_size
        self.model = NN(self.st_size, self.hid, self.act_size)

    def set_trained_network(self, x):
        # Use this block to set the weights AND the biases. Like a real puppet.
        cut0 = self.b0_size
        cut1 = cut0 + self.b2_size
        cut2 = cut1 + self.w0_size

        b0_wts = from_numpy(np.array(x[:cut0]))
        b1_wts = from_numpy(np.array(x[cut0:cut1]))
        w0_wts = from_numpy(np.reshape(x[cut1:cut2], (self.hid, self.st_size)))
        w2_wts = from_numpy(np.reshape(x[cut2:], (self.act_size, self.hid)))

        self.model.set_biases([b0_wts, b1_wts])
        self.model.set_weights([w0_wts, w2_wts])


class RoverWrapper:
    def __init__(self, env, bh):
        self.env = env
        self.p = env.params
        self.agents = self.setup_ag()
        self.st_size = self.env.state_size()
        self.ts = self.p.time_steps
        self.states = np.zeros((self.ts, self.st_size))
        self.act_size = self.env.action_size()
        self.acts = np.zeros((self.ts, self.act_size))
        self.n_obj = self.p.n_poi_types
        self.st_bh_idxs = list(range(8, 17))
        self.act_bh_idxs = [8, 9]

        # This gets the appended portion of the descriptor, which indicates if we're using states (st) or actions (ac)
        if 'auto ' in bh:
            bh = 'auto ' + bh[-2:]
        self.bh_name = bh

    def reset(self):
        self.states = np.zeros((self.ts, self.st_size))
        self.acts = np.zeros((self.ts, self.act_size))
        self.env.reset()

    def setup_ag(self):
        st_size = self.env.state_size()
        hid = lp.hid
        act_size = self.env.action_size()
        ag = []
        for _ in range(self.p.n_agents):
            ag.append(AgPol(st_size, hid, act_size))
        return ag

    def setup(self, x):
        self.reset()
        wts = x
        if self.p.n_agents == 1 and len(x) > 1:
            wts = [x]
        try:
            if len(wts) != self.p.n_agents:
                print(f"given x is of length {len(x)}, should be {self.p.n_agents}")
                return
        except TypeError:
            blerg = 10

        mods = []
        for i, w in enumerate(wts):
            self.agents[i].set_trained_network(w)
            mods.append(self.agents[i].model)
        return mods

    def _evaluate(self, x):
        models = self.setup(x)
        for i in range(self.p.time_steps):
            state = self.env.state()
            self.states[i] = state
            actions = []
            for j, policy in enumerate(models):
                action = policy(state[j]).detach().numpy()
                actions.append(action)
                # THIS ASSUMES THERE IS ONLY ONE AGENT
                self.acts[i] = action
            self.env.action(actions)
        return self.env.G()

    def run_bh(self, x):
        G = self._evaluate(x)
        bh = self.get_bh(self.bh_name)
        return G, bh

    def action_space(self, act_vec, p, n_move):
        idx = np.argmax(act_vec[:-n_move])
        poi_type = int(np.floor(idx / p.n_sensors))
        return np.concatenate(([poi_type], act_vec[-n_move:]))

    def bh_size(self, bh_name):
        # May decide to only use a subset of the state for the behaviors
        # In some domains, part of the state can be binary, which is unhelpful for behaviors
        st_to_use_size = len(self.st_bh_idxs)
        act_to_use_size = len(self.act_bh_idxs)
        sizes = {'avg st': st_to_use_size,                      # Average state
                 'fin st': st_to_use_size,                      # Final state
                 'avg act': act_to_use_size,                  # Average action
                 'fin act': act_to_use_size,                  # Final action
                 'min max st': st_to_use_size * 2,              # Min and max states
                 'min avg max st': st_to_use_size * 3,          # Min, average, and max states
                 'min max act': act_to_use_size * 2,          # Min and max actions
                 'min avg max act': act_to_use_size * 3,      # Min, average, max actions
                 'auto ac': act_to_use_size * self.p.time_steps,
                 'auto st': st_to_use_size * self.p.time_steps    # Auto-encoder will use all states as an input, so we return that to find the behavoir
                 }
        return sizes[bh_name]

    def get_bh(self, bh_name):
        # Sometimes use a subset of the total state
        # For example if the state has binary values or is too big to use as a behavior

        states_to_use = self.states[:, self.st_bh_idxs]
        acts_to_use = self.acts[:, self.act_bh_idxs]
        bhs = {'avg st': np.mean(states_to_use, axis=0),      # Average state
               'fin st': states_to_use[-1],                   # Final state
               'avg act': np.mean(acts_to_use, axis=0),       # Average action
               'fin act': acts_to_use[-1],                    # Final action
               'min max st':                                # Min max states
                   np.concatenate((np.min(states_to_use, axis=0),
                                   np.max(states_to_use, axis=0))),
               'min avg max st':                           # Min, average, max states
                   np.concatenate((np.min(states_to_use, axis=0),
                                   np.mean(states_to_use, axis=0),
                                   np.max(states_to_use, axis=0))),
               'min max act':                               # Min max actions
                   np.concatenate((np.min(acts_to_use, axis=0),
                                   np.max(acts_to_use, axis=0))),
               'min avg max act':                           # Min, average, max actions
                   np.concatenate((np.min(acts_to_use, axis=0),
                                   np.mean(acts_to_use, axis=0),
                                   np.max(acts_to_use, axis=0))),
               'auto ac': np.ndarray.flatten(acts_to_use),
               'auto st': np.ndarray.flatten(states_to_use)}

        return np.nan_to_num(bhs[bh_name])

    def _evaluate_multiple(self, x):
        # n_eval = self.p.n_cf_evals
        # models = self.setup(x)
        # fit_vals = np.zeros((n_eval, self.p.n_poi_types))
        # bh_vals = np.zeros((n_eval, self.p.n_bh))
        # for i in range(n_eval):
        #     self.env.reset()
        #     out_vals = self.run_env(models)
        #     # out_vals = run_env(self.env, models, self.p, use_bh=self.use_bh, vis=self.vis)
        #     if self.use_bh:
        #         fitness, bh = out_vals
        #         fit_vals[i] = fitness
        #         bh_vals[i] = bh
        #     else:
        #         fit_vals[i] = out_vals
        #
        # fits = np.average(fit_vals, axis=0)
        # bhs = np.average(bh_vals, axis=0)
        # if self.use_bh:
        #     return fits, bhs
        # else:
        #     return fits
        pass


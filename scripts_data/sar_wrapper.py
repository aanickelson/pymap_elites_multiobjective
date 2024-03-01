import numpy as np
from pymap_elites_multiobjective.map_elites.neuralnet import NeuralNetwork as NN

import warnings


class SARWrap:
    def __init__(self, env, hid, bh, ts=1000):
        self.env = env
        self.ts = ts
        if 'auto ' in bh:
            bh = 'auto'
        self.bh_name = bh

        self.st_size = self.env.observation_space.shape[0]
        self.states = np.zeros((ts, self.st_size))
        # Change these in case the low or high is inf
        self.st_low = self.env.observation_space.low
        self.st_high = self.env.observation_space.high

        # Found through experimentation, rounded to the nearest 0.1, 0.5, or 1 depending on scale of numbers
        # Actual values found:
        # Min = [0.70525814, -0.19999993, -1.52016221, -2.75772307, -0.97188804, -3.19431965, -5.04843352, -10, -10, -10, -10]
        # Max = [1.75767875, 0.19999226, 0.09941853, 0.11739756, 0.97083518, 5.47730368, 3.31452018, 10., 10., 10., 10]
        if self.env.spec.name == 'mo-hopper-new-rw':
            self.st_low = [0.5, -0.2, -2., -3., -1., -3.5, -5.5, -10., -10., -10., -10.]
            self.st_high = [2., 0.2, 0.2, 0.2, 1., 6., 3.5, 10., 10., 10., 10.]

        # Acutal values found:
        # Min = [-1.20000005, -0.07]
        # Max = [0.44988963, 0.06872208]
        elif self.env.spec.name == 'mo-mountaincarcontinuous-new-rw-v0':
            self.st_low = [-1.3, -0.1]
            self.st_high = [0.5, 0.1]

        # Change these in case the low or high is inf
        else:
            for i in range(len(self.st_low)):
                self.st_low[i] = np.max([self.st_low[i], -50])
                self.st_high[i] = np.min([self.st_high[i], 50])

        # Used for finding the practical min / max of states
        self.raw_st = np.zeros((ts, self.st_size))
        # self.raw_st_low = np.zeros(self.st_size) + 100
        # self.raw_st_high = np.zeros(self.st_size) - 100
        # self.counter = 0

        self.act_size = self.env.action_space.shape[0]
        self.acts = np.zeros((ts, self.act_size))
        self.act_low = self.env.action_space.low
        self.act_high = self.env.action_space.high

        self.n_obj = self.env.unwrapped.reward_dim
        self.model = NN(self.st_size, hid, self.act_size)

    def reset(self):
        self.states = np.zeros((self.ts, self.st_size))
        self.acts = np.zeros((self.ts, self.act_size))
        self.raw_st = np.zeros((self.ts, self.st_size))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.env.reset_custom()
        return self.env.reset()

    def interpolate(self, vec, old_v, new_v):
        old_vals = np.transpose(old_v)
        new_vals = np.transpose(new_v)
        interpolated = np.zeros_like(vec)
        for i, act in enumerate(vec):
            interpolated[i] = np.interp(act, old_vals[i], new_vals[i])
        return interpolated

    def run(self, policy):
        st, _ = self.reset()

        for ts in range(self.ts):
            self.raw_st[ts] = st
            pol_out = policy(st).detach().numpy()

            # Bookkeeping for behaviors
            self.states[ts] = self.interpolate(st, [self.st_low, self.st_high], [[0]*self.st_size, [1]*self.st_size])  # Change to be between [0, 1] for behaviors
            self.acts[ts] = pol_out     # Policy output is between [0,1], which is what we need for behaviors

            # This changes the action to be between the lower and upper bounds for each item in the array
            action = self.interpolate(pol_out, [[0]*self.act_size, [1]*self.act_size], [self.act_low, self.act_high])
            st, vec_reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break

        # Only keep the parts that were filled in, i.e. the number of time steps the sim ran
        self.acts = self.acts[:ts+1]
        self.states = self.states[:ts+1]

        # Used for finding the practical min / max of states
        # raw_st_high_this_time = np.amax(self.raw_st[:ts], axis=0)
        # raw_st_low_this_time = np.amin(self.raw_st[:ts], axis=0)
        # for idx in range(len(self.raw_st_low)):
        #     self.raw_st_high[idx] = np.max((raw_st_high_this_time[idx], self.raw_st_high[idx]))
        #     self.raw_st_low[idx] = np.min((raw_st_low_this_time[idx], self.raw_st_low[idx]))
        # if not self.counter%10000:
        #     print(self.raw_st_low)
        #     print(self.raw_st_high)
        # self.counter += 1

        # Replace nan with 0s
        return np.nan_to_num(self.env.get_wrapper_attr('fin_rw'))

    def run_bh(self, x):
        self.model.set_trained_network(x)
        rw = self.run(self.model)
        bh_vals = self.get_bh(self.bh_name)
        return rw, bh_vals

    def state_size(self):
        return self.st_size

    def action_size(self):
        return self.act_size

    def bh_size(self, bh_name):
        # May decide to only use a subset of the state for the behaviors
        # In some domains, part of the state can be binary, which is unhelpful for behaviors
        st_bh_size = self.env.get_wrapper_attr("st_bh_size")
        sizes = {'avg st': st_bh_size,                      # Average state
                 'fin st': st_bh_size,                      # Final state
                 'avg act': self.act_size,                  # Average action
                 'fin act': self.act_size,                  # Final action
                 'min max st': st_bh_size * 2,              # Min and max states
                 'min avg max st': st_bh_size * 3,          # Min, average, and max states
                 'min max act': self.act_size * 2,          # Min and max actions
                 'min avg max act': self.act_size * 3,      # Min, average, max actions
                 'auto': self.raw_st.size                   # Auto-encoder will use all states as an input
                 }
        return sizes[bh_name]

    def get_bh(self, bh_name):
        # Sometimes use a subset of the total state
        # For example if the state has binary values or is too big to use as a behavior
        st_idx = self.env.get_wrapper_attr('st_bh_idxs')
        states_to_use = self.states[:, st_idx]
        bhs = {'avg st': np.mean(states_to_use, axis=0),      # Average state
               'fin st': states_to_use[-1],                   # Final state
               'avg act': np.mean(self.acts, axis=0),       # Average action
               'fin act': self.acts[-1],                    # Final action
               'min max st':                                # Min max states
                   np.concatenate((np.min(states_to_use, axis=0),
                                   np.max(states_to_use, axis=0))),
               'min avg max st':                           # Min, average, max states
                   np.concatenate((np.min(states_to_use, axis=0),
                                   np.mean(states_to_use, axis=0),
                                   np.max(states_to_use, axis=0))),
               'min max act':                               # Min max actions
                   np.concatenate((np.min(self.acts, axis=0),
                                   np.max(self.acts, axis=0))),
               'min avg max act':                           # Min, average, max actions
                   np.concatenate((np.min(self.acts, axis=0),
                                   np.mean(self.acts, axis=0),
                                   np.max(self.acts, axis=0))),
               'auto': np.ndarray.flatten(self.raw_st)}

        return np.nan_to_num(bhs[bh_name])


class Params:
    def __init__(self):
        self.n_agents = 1



# Cont environments:
# water-reservoir-vo
# mo-mountaincarcontinuous-v0
# mo-lunar-lander-v2
# mo-hopper-v4
# mo-halfcheetah-v4

# Helpful links
# https://mo-gymnasium.farama.org/environments/water-reservoir/
# https://gymnasium.farama.org/api/wrappers/reward_wrappers/
# https://pymoo.org/interface/minimize.html

# Paper with a lot of the implementaiton details
# https://openreview.net/forum?id=AwWaBXLIJE

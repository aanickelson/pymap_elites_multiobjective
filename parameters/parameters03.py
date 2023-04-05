import itertools
from random import random
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent


def init_fn(n):
    return n + random(), n + random()


class Parameters:
    param_idx = 3  # Makes it easy to differentiate results by parameter set

    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 5.0
    map_size = 20

    # poi_pos = [[10, 10], [20, 10], [10, 20], [20, 20]]
    poi_pos = list(itertools.product([4, 8, 12, 16], repeat=2))
    n_pois = len(poi_pos)
    poi_class = [linear_poi, tanh_poi] * int(n_pois/2)  # + [tanh_poi] * int(n_pois/2)
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos = [init_fn(10) for i in range(n_agents)]

    interact_range = 2.0
    n_sensors = 4


if __name__ == '__main__':
    p = Parameters()
    p.n_sensors = 8

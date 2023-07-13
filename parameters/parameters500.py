import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 500
    n_agents = 2
    battery = 30
    time_steps = 50
    speed = 2.0
    map_size = 20
    counter = 0
    cf_bh = False
    counter_move = False

    poi_visit = False    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs = []

    poi_pos = [[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos = [[9.011948381043402, 10.401692868409176], [10.011948381043402, 9.401692868409176]]

    interact_range = 2.0
    n_sensors = 4


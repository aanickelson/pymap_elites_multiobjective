import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 502
    n_agents = 2
    battery = 30
    time_steps = 50
    speed = 2.0
    n_cf_evals = 10  # Number of times to rerun with different cf configurations
    map_size = 20
    counter = 1
    cf_bh = False
    counter_move = True

    poi_visit = True    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs = [[2.1768243236608047, 2.215317159566301], [2.8685358828024254, 18.693215873656115]]

    poi_pos = [[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos = [[10.186212215290968, 9.376711740589133], [10.186212215290968, 9.376711740589133]]

    interact_range = 2.0
    n_sensors = 4


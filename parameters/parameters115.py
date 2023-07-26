import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 115
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 2.0
    n_cf_evals = 10  # Number of times to rerun with different cf configurations 
    map_size = 20
    counter = 5
    cf_bh = True
    counter_move = True
    ag_in_st = True

    poi_visit = True    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[1.6124569426178472, 1.26740194760333], [2.6654891343196288, 17.965869965604213], [17.69824741525216, 1.1143531139389755], [18.303614787775224, 17.141376009938995], [2.736498845775429, 2.4661656421228453]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[9.02518917475535, 9.994396567876612]]

    interact_range = 2.0
    n_sensors = 4


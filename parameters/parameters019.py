import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 19
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 2.0
    map_size = 20
    counter = 9
    cf_bh = True
    counter_move = True

    poi_visit = False    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[2.8118204951186825, 1.6258874345028238], [1.639692175903954, 17.023134810633913], [17.800688806024773, 2.6775084109127834], [18.406813396141203, 17.539408940015797], [2.791917700329263, 1.5327931901076952], [1.2442971671350065, 18.855343452012928], [18.278712785722416, 1.251332396559901], [18.062383189449633, 18.50661297728221], [2.4940949836785546, 1.34984146264797]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[9.011948381043402, 10.401692868409176]]

    interact_range = 2.0
    n_sensors = 4


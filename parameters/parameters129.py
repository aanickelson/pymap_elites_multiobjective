import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 119
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 2.0
    map_size = 20
    counter = 9
    cf_bh = False
    counter_move = True

    poi_visit = True    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[2.068160992414275, 1.6610562051050872], [2.093070747038758, 17.291417684939834], [18.275338054359448, 2.8845992601204626], [18.12062422486274, 17.031118699124104], [2.6482916329556137, 1.488628308195248], [2.4136554388060842, 18.64136986778534], [17.190180892507936, 2.7952489308734294], [18.283838972294028, 18.480606941326247], [1.986861843319424, 1.7841078469895355]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[9.011948381043402, 10.401692868409176]]

    interact_range = 2.0
    n_sensors = 4


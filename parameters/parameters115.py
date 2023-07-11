import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 115
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 2.0
    map_size = 20
    counter = 5
    counter_move = True

    poi_visit = True    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[2.030930255193427, 2.136945374389551], [1.8920292811819026, 18.84861165665756], [17.489960831608773, 1.6026690583615772], [18.98212894396924, 18.055156942029537], [2.483347587689261, 2.018788566511268]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[10.802552714796493, 9.881289509824887]]

    interact_range = 2.0
    n_sensors = 4


import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 9
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 2.0
    map_size = 20
    counter = 9
    cf_bh = True
    counter_move = False
    ag_in_st = True

    poi_visit = False    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[2.188792249192018, 2.119203013555884], [2.7074336854957464, 17.907488472102685], [18.107963816066633, 2.853925575142264], [18.517363903660907, 18.60662758259549], [2.687079569048702, 1.8734267500552098], [1.542475514947339, 17.040684730018107], [17.53544869922184, 2.2152306551981775], [18.99719283801884, 17.65392133523175], [2.949948856245104, 2.8351721325150647]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[10.326010252651205, 10.631393351431132]]

    interact_range = 2.0
    n_sensors = 4


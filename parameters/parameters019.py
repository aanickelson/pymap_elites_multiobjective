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
    ag_in_st = True

    poi_visit = False    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[2.5005044569220125, 2.031095706448536], [1.1239994952353916, 18.84087131259896], [18.873083057253474, 1.4073386567294546], [18.476933461364535, 18.546133391325593], [2.004635840448488, 2.629652245471383], [1.2373377817168387, 18.700036557021598], [17.486134525351673, 2.175409026272135], [18.867382821446885, 17.038117634940377], [2.0097426761235706, 1.852633861677124]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[10.326010252651205, 10.631393351431132]]

    interact_range = 2.0
    n_sensors = 4


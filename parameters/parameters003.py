import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 3
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 2.0
    n_cf_evals = 10  # Number of times to rerun with different cf configurations 
    map_size = 20
    counter = 3
    cf_bh = True
    counter_move = False
    ag_in_st = True

    poi_visit = False    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[2.8459157107242863, 1.3261134023456793], [1.4317280073397634, 18.67659635108975], [18.0654623842372, 2.50332735778382]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[9.02518917475535, 9.994396567876612]]

    interact_range = 2.0
    n_sensors = 4


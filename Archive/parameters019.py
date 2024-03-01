import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 19
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 2.0
    n_cf_evals = 10  # Number of times to rerun with different cf configurations 
    map_size = 20
    counter = 9
    cf_bh = True
    counter_move = True
    ag_in_st = True

    poi_visit = False    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[2.0081133082254077, 1.1153974653693894], [1.5446565555687668, 17.436337215755767], [18.527296879791994, 1.2529271429274], [18.686538946088294, 17.77078135568296], [1.987423001941535, 1.4446547580903113], [2.1717305392583297, 18.758390197634142], [17.81299006253653, 1.690721217351815], [17.446981602996146, 17.685383889949623], [1.5389329319490428, 2.181488736538638]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[9.02518917475535, 9.994396567876612]]

    interact_range = 2.0
    n_sensors = 4


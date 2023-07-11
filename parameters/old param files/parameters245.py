import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 245
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 5.0
    map_size = 20
    counter = 5
    counter_move = True

    poi_visit = False    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[1.4150664490773615, 2.4628930288535713], [1.801906932973108, 17.235904067243794], [18.3752758768731, 2.917397706316743], [18.10524020392819, 18.03044068323039], [2.0312108406948597, 1.0207516645541816]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[10.181812237245886, 9.856613026319122]]

    interact_range = 2.0
    n_sensors = 4


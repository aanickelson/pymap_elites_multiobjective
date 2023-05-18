import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 349
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 5.0
    map_size = 20
    counter = 9
    counter_move = True

    poi_visit = True    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[1.6686162608359643, 2.8862073870670537], [2.976296766269445, 18.27217372062145], [17.342446727134828, 2.803609902297837], [18.446555235144135, 18.550352997054564], [2.1100716704641123, 2.7401139711236295], [1.3072284988815255, 18.911412158463037], [18.71953794874432, 1.4437970810979712], [18.51693596355568, 17.636646948415937], [1.1031354913969784, 2.003268328644541]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[10.181812237245886, 9.856613026319122]]

    interact_range = 2.0
    n_sensors = 4


import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 249
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 5.0
    map_size = 20
    counter = 9
    counter_move = True

    poi_visit = False    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[1.395163096160281, 2.370081703198233], [1.1715416214164835, 18.202894741043593], [18.812007582214633, 1.17874769900581], [18.897265688972425, 18.236818587584242], [2.463887421445733, 2.7227700038512865], [2.73029693664275, 17.653848633532824], [17.66135660634898, 2.265520698959428], [17.366003548886894, 18.46380558262707], [2.215558075866809, 2.9144329716830275]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[10.181812237245886, 9.856613026319122]]

    interact_range = 2.0
    n_sensors = 4


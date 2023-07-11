import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 5
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 2.0
    map_size = 20
    counter = 5
    counter_move = False

    poi_visit = False    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[2.004027435209692, 1.925892982922773], [1.6267884219749609, 17.985086257762475], [17.332431158127562, 2.148470949494315], [18.693442151469444, 17.097147402063083], [1.4119427861496217, 2.8166057512045395]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[10.802552714796493, 9.881289509824887]]

    interact_range = 2.0
    n_sensors = 4


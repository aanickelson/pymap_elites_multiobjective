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
    counter_move = False

    poi_visit = False    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[1.509761273614157, 1.7735165909910253], [2.7869110273928066, 17.28087501983061], [18.882501888129323, 2.8876652301530026], [18.161861153920036, 18.146657807179125], [1.8991529125702706, 1.8645933197843445], [2.919675359153912, 18.711426850283736], [18.503677226868653, 1.9968117291428036], [17.778936935308312, 18.552213492834607], [1.0105288672606902, 1.8109921474218746]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[10.802552714796493, 9.881289509824887]]

    interact_range = 2.0
    n_sensors = 4


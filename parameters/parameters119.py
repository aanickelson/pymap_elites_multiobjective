import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 119
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 2.0
    map_size = 20
    counter = 9
    counter_move = True

    poi_visit = True    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[2.288111006566067, 1.3160475211180451], [1.4019870301718478, 18.63813659147973], [17.238607364175724, 1.4424079788504676], [18.952018945408327, 18.943041595213348], [2.1349633053186574, 2.435836570771217], [1.189071125705205, 17.54499811345521], [17.107165071207938, 2.836538470360727], [17.76314869251295, 17.161310655529327], [1.945476804327782, 2.4896183551134854]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[10.802552714796493, 9.881289509824887]]

    interact_range = 2.0
    n_sensors = 4


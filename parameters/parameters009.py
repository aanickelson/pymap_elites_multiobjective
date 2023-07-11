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

    poi_visit = False    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[2.2091487826654057, 1.4306164760706992], [1.5383048318025399, 17.053720111690385], [17.966038010458, 2.6333020007233374], [18.060211829187644, 17.362888238789115], [1.4836731466101118, 2.9891966800745395], [1.9378639547148007, 17.22132520793048], [17.338586554080873, 2.724546173968041], [18.30658366925456, 17.90849258529425], [1.168990593362226, 2.2884250851179493]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[9.011948381043402, 10.401692868409176]]

    interact_range = 2.0
    n_sensors = 4


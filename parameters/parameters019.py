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
    counter_move = True

    poi_visit = False    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[[2.117176616718963, 1.5570059162220236], [1.2305820505622314, 17.037955462481467], [17.525095160028968, 1.1895636648136465], [17.523450057045256, 18.456319454763246], [2.138469217372256, 2.7343294326374643], [2.7213083474761834, 17.763731516110468], [18.025898483879757, 2.7362624438762007], [18.719918704472533, 18.343158325728183], [1.7780069141161312, 2.700157409851494]]

    poi_pos =[[1.0, 5.5], [1.0, 10.0], [1.0, 19.0], [5.5, 1.0], [5.5, 10.0], [5.5, 14.5], [10.0, 1.0], [10.0, 5.5], [10.0, 14.5], [10.0, 19.0], [14.5, 5.5], [14.5, 10.0], [14.5, 19.0], [19.0, 1.0], [19.0, 10.0], [19.0, 14.5]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[10.802552714796493, 9.881289509824887]]

    interact_range = 2.0
    n_sensors = 4


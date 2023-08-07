import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 310
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 2.0
    n_cf_evals = 10  # Number of times to rerun with different cf configurations 
    map_size = 20
    counter = 0
    cf_bh = False
    ag_in_st = True

    counter_move = True
    poi_visit = True    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    counter_locs =[]

    poi_pos =[[15.025451975691034, 9.4533297424305], [14.937885332453975, 11.356006980312857], [13.290020419824964, 14.188384769614686], [13.032739746847351, 15.65345996037452], [9.944640177015746, 15.296740005698469], [8.491962261848833, 16.40762533385623], [5.257701887432509, 14.641586042446093], [4.382708097653762, 13.288459406203923], [3.627165233873834, 10.06119401098483], [4.61900482573196, 8.611394916914733], [5.25656334661754, 4.803631193095731], [7.181197760641557, 4.441145582313218], [9.461685118839842, 3.287079804244346], [11.487651332371012, 3.540197608342005], [13.866870126128056, 5.222514951433874], [14.549376648224847, 7.790953529799991]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[9.616698327362716, 9.478654587705297]]

    interact_range = 2.0
    n_sensors = 4


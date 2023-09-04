import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 101103
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 2.0
    n_cf_evals = 10  # Number of times to rerun with different cf configurations 
    map_size = 20
    counter = 3
    counter_locs =[[2.980435759172686, 2.0485121174799614], [1.259774108724047, 18.162961372875003], [17.53363938756863, 2.2518748184309665]]

    counter_move = False
    poi_visit = True    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    ag_in_st = True
    cf_bh = False

    poi_pos =[[15.391039684524003, 10.37932221206273], [15.45144203833575, 11.755074078471255], [13.355935843022364, 14.889989968729646], [13.233921506075543, 15.213311285326382], [9.201751930124022, 16.92991385157506], [6.8137747832717395, 15.437722327484943], [5.759393595249292, 14.774683314343479], [3.6703737289491363, 11.745317670717245], [4.920088790912753, 9.035198402932817], [5.093268850307474, 7.606784465066059], [6.552073790520998, 5.770451300971435], [8.48739800284285, 3.707829689873673], [10.65930447466113, 4.704472581421585], [12.238561083376307, 5.09531451685917], [14.486891375652583, 6.528249624651417], [15.982828664555628, 6.88346239078825]]
    n_pois = 16
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[10.093892985948093, 9.922354810532951]]

    interact_range = 2.0
    n_sensors = 4



def string_to_save(i, n_cf, cf_bh, map_size, move, poi, cf_locs, poi_locs, n_poi, agent_locs, ag_st):
    s = ""
    s += "import numpy as np\n"
    s += "from AIC.poi import tanh_poi, linear_poi\n"
    s += "from AIC.agent import agent\n\n"
    s += "class Parameters:\n\n"

    s += f"    param_idx = {i}\n"
    s += f"    n_agents = 1\n"
    s += f"    battery = 30\n"
    s += f"    time_steps = 50\n"
    s += f"    speed = 2.0\n"
    s += f"    n_cf_evals = 10  # Number of times to rerun with different cf configurations \n"
    s += f"    map_size = {map_size}\n"
    s += f"    counter = {n_cf}\n"
    s += f"    counter_locs =" + str([l for l in cf_locs]) + "\n\n"

    s += f"    counter_move = {bool(move)}\n"
    s += f"    poi_visit = {bool(poi)}    # Flag to determine if agent impacts POI completeness, but NOT the rewards\n"
    s += f'    ag_in_st = {bool(ag_st)}\n'
    s += f"    cf_bh = {bool(cf_bh)}\n\n"

    s += f"    poi_pos =" + str(poi_locs) + "\n"
    s += f"    n_pois = {n_poi}\n"
    s += f"    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]\n"
    s += f"    n_poi_types = 2\n\n"

    s += f"    agent_class = [agent] * n_agents\n"
    s += f"    agent_pos =" + str(agent_locs) + "\n\n"

    s += f"    interact_range = 2.0\n"
    s += f"    n_sensors = 4\n\n"

    return s


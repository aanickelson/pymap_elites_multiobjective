import numpy as np
from AIC.poi import tanh_poi, linear_poi
from AIC.agent import agent

class Parameters:

    param_idx = 211109
    n_agents = 1
    battery = 30
    time_steps = 50
    speed = 2.0
    n_cf_evals = 10  # Number of times to rerun with different cf configurations 
    map_size = 20
    counter = 9
    counter_locs =[[2.3990255056638095, 2.1797072930043866], [2.54682636151247, 18.467086568500605], [18.64044743471486, 1.1929607834257137], [17.636974000483402, 18.10439929713877], [2.1728440532259743, 2.291830141882104], [2.5933537773556243, 18.949258043018983], [17.745975753951043, 1.0396743564826878], [18.712465114725163, 17.22370074745591], [2.139017901213427, 2.7823012529759388]]

    counter_move = True
    poi_visit = True    # Flag to determine if agent impacts POI completeness, but NOT the rewards
    ag_in_st = True
    cf_bh = False

    poi_pos =[[15.05668124674774, 10.178608539294995], [16.84626717170053, 11.273545855098337], [14.776507041201697, 12.364962604565182], [15.278117609297095, 12.286544683465566], [15.65583141269124, 13.207068898069107], [14.833470937317347, 14.265787042772216], [13.771551522131091, 15.467613716296961], [12.005100596359704, 14.99590265355655], [11.786681127119222, 16.7048410387353], [11.534521037792324, 15.65288500269223], [10.087900682437098, 16.942808943841932], [9.594876339930844, 15.36709992015199], [8.93226885466903, 16.177496588667555], [6.756979120109747, 15.39036795397909], [7.200777144632779, 13.862888361735248], [6.2505274166041005, 14.191990859280246], [5.8415702970596355, 14.141028117524135], [5.092864881979577, 13.052134843369172], [3.7948491579989176, 11.657092246486327], [4.113752455301939, 10.022927034851637], [4.412413259947531, 10.199994173121869], [3.2211476606805602, 8.634484213970433], [3.45394783591121, 7.541900275851414], [4.254135571582592, 7.153799640887869], [4.502226225838852, 6.9131564461457184], [6.1366892298322355, 5.702271192126751], [5.8453407670164195, 4.413762175832955], [7.337500309490288, 4.6617718974945985], [8.481441344032831, 3.4112144156425313], [8.416971059762998, 3.291000863707817], [10.071033678713661, 3.370854174375368], [11.349858808283873, 4.927593966347738], [11.11050428295655, 4.343129318923005], [12.951573190351022, 3.7337145237496143], [13.26432756464299, 5.388936791867199], [13.703536133917174, 6.3403094342968895], [14.474952215024265, 6.286829721027711], [15.955100740944715, 7.97948799186679], [16.258414861339812, 7.362003395894993], [15.359104798556587, 8.455228464379344]]
    n_pois = 40
    poi_class = ([linear_poi, tanh_poi] * int(np.ceil(n_pois / 2)))[:n_pois]
    n_poi_types = 2

    agent_class = [agent] * n_agents
    agent_pos =[[9.68727103948704, 9.895469950971664]]

    interact_range = 2.0
    n_sensors = 4


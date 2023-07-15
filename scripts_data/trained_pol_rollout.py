from AIC.aic import aic
import pymap_elites_multiobjective.parameters as Params
from evo_playground.support.rover_wrapper import RoverWrapper
import numpy as np


def get_weights(fpath):
    data = np.loadtxt(fpath)
    # if CFs, then use 12. If no cfs, use 14. Because hard coding is easy. kinda.
    return data


def init_setup(params):
    params.n_agents = 2
    en = aic(params)
    wra = RoverWrapper(en, params)
    wra.vis = False
    wra.use_bh = False
    return wra


def main(wrap, file_data, pnums):
    # To find good policies
    # finding_pols = np.zeros((len(file_data), 3))
    # finding_pols[:, 0] = [i for i in range(len(file_data))]
    # finding_pols[:, 1:] = file_data[:, :2]

    og_sum = np.zeros(2)
    w = file_data[pnums]
    og_sum += np.sum(w[:, :2], axis=0)
    all_wts = w[:, 14:]

    # This will be 12 if the behavior space is size 5; 14 if size 6
    # 2 objectives + bh description + centroid (2 + 5/6 + 5/6)
    fit = wrap._evaluate(all_wts)
    # if sum(fit) < 0.005:
    #     wrap.vis = True
    #     wrap.evaluate([wts])
    #     exit()
    # wrap.vis = False
    # print(og_sum)
    # print(fit)
    percent_captured = (sum(abs(og_sum - fit)) / sum(og_sum))
    # print(percent_captured)
    return percent_captured


if __name__ == '__main__':
    pth = '/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data/518_20230713_151107/111_run0/weights_100000.dat'
    # 111
    # pol_nums = [69, 70, 71, 283, 419, 420, 1136]
    pol_nums = [1085, 1911, 28, 8, 602,   # g1
                1321, 1415, 2, 1125,      # g2
                489, 232, 52, 3]    # g1 & 2
    dict = {500: [], 501: [], 502: []}
    ag_pos = [[[10.186212215290968, 9.376711740589133], [2.8685358828024254, 18.693215873656115]]]
    for ag in ag_pos:
        for blerg in range(20):
            # print(blerg)
            for par in [Params.p500, Params.p501, Params.p502]:
                if ag:
                    par.agent_pos = ag
                pols = np.random.choice(pol_nums, size=2)
                w = init_setup(par)
                f = get_weights(pth)
                pct = main(w, f, pols)
                dict[par.param_idx].append(pct)
        print(ag)

        for key, value in dict.items():
            print(key, np.average(value), np.std(value))

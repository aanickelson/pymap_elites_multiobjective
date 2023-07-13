from AIC.aic import aic
from pymap_elites_multiobjective.parameters.parameters000 import Parameters as params
from evo_playground.learning.rover_wrapper import RoverWrapper
import numpy as np


def get_weights(fpath):
    data = np.loadtxt(fpath)
    # if CFs, then use 12. If no cfs, use 14. Because hard coding is easy. kinda.
    return data


def init_setup(pol_idxs):

    data_path = '/home/anna/PycharmProjects/pymap_elites_multiobjective/scripts_data/data/015_20230711_151938/501_119_run0_min/weights_100000.dat'
    en = aic(params)
    wra = RoverWrapper(en, params)
    wra.vis = False
    wra.use_bh = False

    file_dat = get_weights(data_path)
    return wra, file_dat


def main(wrap, file_data, pnums):

    i = pnums[0]
    w = file_data[i]
    og = w[:2]

    # This will be 12 if the behavior space is size 5; 14 if size 6
    # 2 objectives + bh description + centroid (2 + 5/6 + 5/6)
    wts = w[12:]
    fit = wrap.evaluate([wts])
    # print(og)
    print(fit)
    # print(fit - (og))


if __name__ == '__main__':
    pol_nums = [273, 84, 61, 136, 22, 466, 409, 98, 282, 149, 63, 716, 717]
    for pol in pol_nums:
        w, f = init_setup([pol])
        # for _ in range(100):
        main(w, f, [pol])

from AIC.aic import aic
from pymap_elites_multiobjective.parameters.parameters000 import Parameters as params
from evo_playground.learning.rover_wrapper import RoverWrapper
import numpy as np


def get_weights(fpath):
    data = np.loadtxt(fpath)
    # if CFs, then use 12. If no cfs, use 14. Because hard coding is easy. kinda.
    return data

def init_setup(pol_idxs):

    data_path = '/home/anna/PycharmProjects/pymap_elites_multiobjective/scripts_data/data/002_20230710_222455/010_run5/weights_200.dat'
    params.speed = 2.0
    params.n_agents = len(pol_idxs)
    en = aic(params)
    wra = RoverWrapper(en, params)
    wra.vis = False
    wra.use_bh = False

    file_dat = get_weights(data_path)
    return en, wra, file_dat

def main(env, wrap, file_data):

    i = 0
    w = file_data[i]
    og = w[:2]
    wts = w[14:]
    fit = wrap.evaluate(wts)
    x = fit
    # print(og)
    # print(x)
    print(abs(x - og))


if __name__ == '__main__':
    pol_nums = []
    e, w, f = init_setup(pol_nums)
    # for _ in range(100):
    main(e, w, f)

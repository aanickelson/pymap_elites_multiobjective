from AIC.aic import aic
from pymap_elites_multiobjective.parameters.parameters345 import Parameters as params
from cvt_aic import RoverWrapper
import numpy as np


def get_weights(fpath):
    data = np.loadtxt(fpath)[6]
    return data[5:]

if __name__ == '__main__':

    data_path = '/home/anna/PycharmProjects/pymap_elites_multiobjective/scripts_data/data/510_20230626_153034/345_run0/weights_200000.dat'
    wts = get_weights(data_path)
    env = aic(params)
    wrap = RoverWrapper(env, params)
    wrap.vis = True
    wrap.evaluate(wts)


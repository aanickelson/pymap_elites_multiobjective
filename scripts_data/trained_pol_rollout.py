from AIC.aic import aic
from pymap_elites_multiobjective.parameters.parameters345 import Parameters as params
from cvt_aic import RoverWrapper
import numpy as np
from matplotlib import pyplot as plt

def get_weights(fpath):
    data = np.loadtxt(fpath)[97]
    return data[5:]

if __name__ == '__main__':
    data_path = '/home/anna/PycharmProjects/pymap_elites_multiobjective/scripts_data/data/511_20230626_172709/345_run5/weights_200000.dat'
    wts = get_weights(data_path)
    params.speed = 2.0
    env = aic(params)

    x = np.zeros((500, 2))
    for i in range(500):
        wrap = RoverWrapper(env, params)
        # wrap.vis = True
        wrap.vis = False
        fit, beh = wrap.evaluate(wts)
        x[i] = fit
        # print(fit, beh)

    print(x)
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()

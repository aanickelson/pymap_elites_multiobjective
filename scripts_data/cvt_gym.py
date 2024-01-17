# only required to run python3 Archive/cvt_rastrigin.py
import random
import numpy.random
from time import time
from datetime import datetime
from os import path, getcwd
import multiprocessing
import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics

import pymap_elites_multiobjective.map_elites.cvt as cvt_me
import pymap_elites_multiobjective.map_elites.cvt_auto_encoder as cvt_auto_encoder
from pymap_elites_multiobjective.cvt_params.mome_default_params import default_params
from pymap_elites_multiobjective.cvt_params.gym_params_0000 import Parameters
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp
import pymap_elites_multiobjective.scripts_data.often_used as oft
from pymap_elites_multiobjective.scripts_data.sar_wrapper import SARWrap


def main(setup):
    [env_p, cvt_p, filepath, env_nm, bh_name, stat_num] = setup
    print(f"main has begun for {bh_name} - {stat_num}")
    numpy.random.seed(stat_num + random.randint(0, 10000))
    moo_gym_env = MORecordEpisodeStatistics(mo_gym.make(env_nm), gamma=0.99)
    # eval_env = mo_gym.make(gym_env)
    if env_nm == "mo-hopper-new-rw-v4":
        ts = 200
    elif env_nm == '"mo-mountaincarcontinuous-new-rw-v0"':
        ts = 200
    else:
        ts = 1000

    wrap = SARWrap(moo_gym_env, lp.hid, bh_name, ts)
    # Dimension of x to be tested is the sum of the sizes of the weights vectors and bias vectors
    wts_dim = ((wrap.st_size * lp.hid)      # Layer 0 size
               + (lp.hid * wrap.act_size)   # Layer 1 size
               + lp.hid                     # Bias 0 size
               + wrap.act_size)             # Bias 1 size
    if bh_name == "auto so" or bh_name == 'auto mo':
        n_behaviors = 2
        multiobjective = bh_name == 'auto mo'
        archive = cvt_auto_encoder.compute(n_behaviors, wts_dim, wrap, n_niches=px['n_niches'], max_evals=cvt_p["evals"],
                                 log_file=open('cvt.dat', 'w'), params=cvt_p, data_fname=filepath, multiobj=multiobjective)

    else:
        n_behaviors = wrap.bh_size(wrap.bh_name)
        archive = cvt_me.compute(n_behaviors, wts_dim, wrap.run_bh, n_niches=px['n_niches'], max_evals=cvt_p["evals"],
                                 log_file=open('cvt.dat', 'w'), params=cvt_p, data_fname=filepath)


def multiprocess_main(batch_for_multi):
    cpus = multiprocessing.cpu_count() - 1
    # cpus = 4
    with multiprocessing.Pool(processes=cpus) as pool:
        pool.map(main, batch_for_multi)


if __name__ == '__main__':
    px = default_params.copy()
    p = Parameters

    # DEBUGGING VALS:
    # px["batch_size"] = 100
    # px["dump_period"] = 100
    # px['n_niches'] = 1000
    # px['evals'] = 200
    px['evals'] = 100000

    bh_options_hop = ['auto so', 'auto mo', 'avg st', 'fin st', 'avg act', 'fin act', 'min max st', 'min avg max st', 'min max act', 'min avg max act']
    # bh_options_hop = ['auto so', 'auto mo']
    # Action in mountain car is 1d, so not very useful as a behavior descriptor
    bh_options_mt = ['auto so', 'auto mo', 'avg st', 'fin st', 'min max st', 'min avg max st', 'min max act', 'min avg max act']
    # bh_options_mt = ['auto mo', 'auto so']
    env_info = [["mo-hopper-new-rw-v4", 'hopper', bh_options_hop],
                ["mo-mountaincarcontinuous-new-rw-v0", 'mountain', bh_options_mt]]
    # env_info = [["mo-mountaincarcontinuous-new-rw-v0", 'mountain', bh_options_mt]]
    # env_info = [['mo-lunar-lander-continuous-new-rw-v2', 'lander', bh_options_mt]]
    # env_info = [["mo-hopper-new-rw-v4", 'hopper', bh_options_hop]]

    # bh_options = ['avg st']  #, 'fin st', 'avg act', 'fin act', 'min avg max act']
    # bh_options = ['min avg max act', 'fin act']

    # lp.n_stat_runs = 10
    lp.n_stat_runs = 10

    batch = []
    for env_name, env_shorthand, bh_options in env_info:

        base_path = path.join(getcwd(), 'data_gym', env_shorthand)
        oft.make_a_directory(base_path)

        now = datetime.now()
        now_str = now.strftime("_%Y%m%d_%H%M%S")
        dirpath = oft.get_unique_fname(base_path, now_str)
        oft.make_a_directory(dirpath)

        for b in bh_options:
            for i in range(lp.n_stat_runs):
                filepath = path.join(dirpath, f'{p.param_idx:03d}_{b}_run{i}')
                oft.make_a_directory(filepath)
                batch.append([p, px, filepath, env_name, b, i])

    # Use this one to multiprocess
    multiprocess_main(batch)

    # This runs a single experiment / setup at a time for debugging
    # px["parallel"] = True
    # main(batch[0])

    # This runs them one at a time
    # for b in batch:
    #     main(b)

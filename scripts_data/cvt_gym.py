# only required to run python3 Archive/cvt_rastrigin.py
import random
import numpy.random
from time import time
from datetime import datetime
from os import path, getcwd
import multiprocessing
import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics


from AIC.aic import aic as Domain
import pymap_elites_multiobjective.map_elites.cvt as cvt_me
from pymap_elites_multiobjective.cvt_params.mome_default_params import default_params
from pymap_elites_multiobjective.cvt_params.gym_params_0000 import Parameters
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp
import pymap_elites_multiobjective.scripts_data.often_used as oft
from evo_playground.test_morl.sar_wrapper import SARWrap


def main(setup):
    [env_p, cvt_p, filepath, env_nm, bh_name, stat_num] = setup
    print(f"main has begun for {bh_name} - {stat_num}")
    numpy.random.seed(stat_num + random.randint(0, 10000))
    moo_gym_env = MORecordEpisodeStatistics(mo_gym.make(env_nm), gamma=0.99)
    # eval_env = mo_gym.make(gym_env)
    wrap = SARWrap(moo_gym_env, lp.hid, bh_name)
    # Dimension of x to be tested is the sum of the sizes of the weights vectors and bias vectors
    wts_dim = ((wrap.st_size * lp.hid)      # Layer 0 size
               + (lp.hid * wrap.act_size)   # Layer 1 size
               + lp.hid                     # Bias 0 size
               + wrap.act_size)             # Bias 1 size

    n_behaviors = wrap.bh_size(wrap.bh_name)
    start = time()
    archive = cvt_me.compute(n_behaviors, wts_dim, wrap.run_bh, n_niches=px['n_niches'], max_evals=cvt_p["evals"],
                             log_file=open('cvt.dat', 'w'), params=cvt_p, data_fname=filepath)
    tot_time = time() - start
    with open(filepath + '_time.txt', 'w') as f:
        f.write(str(tot_time))


def multiprocess_main(batch_for_multi):
    cpus = multiprocessing.cpu_count() - 1
    # cpus = 4
    with multiprocessing.Pool(processes=cpus) as pool:
        pool.map(main, batch_for_multi)


if __name__ == '__main__':
    px = default_params.copy()
    p = Parameters
    env_name = "mo-lunar-lander-continuous-v2"
    env_shorthand = 'lander'

    # DEBUGGING VALS:
    # px["batch_size"] = 10
    # px["dump_period"] = 100
    # px['n_niches'] = 100
    # px['evals'] = 200

    base_path = path.join(getcwd(), 'data_gym', env_shorthand)
    oft.make_a_directory(base_path)

    now = datetime.now()
    now_str = now.strftime("_%Y%m%d_%H%M%S")
    dirpath = oft.get_unique_fname(base_path, now_str)
    oft.make_a_directory(dirpath)
    batch = []

    bh_options = ['avg st', 'fin st', 'avg act', 'fin act', 'min avg max act']
    # bh_options = ['avg st']  #, 'fin st', 'avg act', 'fin act', 'min avg max act']

    lp.n_stat_runs = 5
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

# only required to run python3 Archive/cvt_rastrigin.py
import random
import numpy.random
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
    if 'auto' in bh_name:
        n_behaviors = 2
        # Flag that indicates if it is single or multi-objective
        multiobjective = 'auto mo' in bh_name
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

    px['evals'] = 100000

    # DEBUGGING VALS:
    # px["batch_size"] = 10
    # px["dump_period"] = 10
    # px['n_niches'] = 1000
    # px['evals'] = 20
    # px['evals'] = 100

    # Behaviors you want to test in the hopper environment
    # bh_options_hop = ['auto so', 'auto mo', 'avg st', 'fin st', 'avg act', 'fin act', 'min max st', 'min avg max st', 'min max act', 'min avg max act']
    bh_options_hop = ['auto so', 'auto mo']

    # Behaviors you want to test in the mountain car environment
    # Action in mountain car is 1d, so not very useful as a behavior descriptor
    bh_options_mt = ['auto so', 'auto mo', 'avg st', 'fin st', 'min max st', 'min avg max st', 'min max act', 'min avg max act']
    # bh_options_mt = ['auto mo', 'auto so']

    # Pick which environments you want to test
    env_info = [["mo-hopper-new-rw-v4", 'hopper', bh_options_hop],]
                # ["mo-mountaincarcontinuous-new-rw-v0", 'mountain', bh_options_mt]]

    # How many stat runs per behavior x environment
    lp.n_stat_runs = 10

    ######################################################
    # You shouldn't need to change anything below here
    # Unless you want to run only one experiment for debugging purposes (go to the end comment block)
    ######################################################

    p = Parameters

    # Set up directories for data
    data_fpath = path.join(getcwd(), 'data')
    oft.make_a_directory(data_fpath)
    batch = []

    # Set up batch parameters to run
    for env_name, env_shorthand, bh_options in env_info:

        # More sub-directories for data
        base_path = path.join(data_fpath, env_shorthand)
        oft.make_a_directory(base_path)

        now = datetime.now()
        now_str = now.strftime("_%Y%m%d_%H%M%S")
        dirpath = oft.get_unique_fname(base_path, now_str)
        oft.make_a_directory(dirpath)

        # Batch parameters
        for b in bh_options:
            for i in range(lp.n_stat_runs):
                filepath = path.join(dirpath, f'{p.param_idx}_{b}_run{i:02d}')
                oft.make_a_directory(filepath)
                batch.append([p, px, filepath, env_name, b, i])

    ######################################################
    # Comment / uncomment the sections below depending on if you want to use multiprocessing
    ######################################################

    # Use this one to multiprocess
    multiprocess_main(batch)

    # This runs a single experiment for debugging
    # 'parallel' flag determines if you test the environments in parallel, i.e. the policy evaluation
    # cannot be combined with multiprocessing
    # px["parallel"] = True
    # main(batch[0])

    # This runs one at a time for debugging
    # px["parallel"] = True
    # for b in batch:
    #     main(b)

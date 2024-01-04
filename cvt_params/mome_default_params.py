default_params = \
    {
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to paralleliez
        "batch_size": 100,
        # proportion of niches to be filled before starting
        "random_init": 0.001,
        # batch for random initialization
        "random_init_batch": 100,
        # when to write results (one generation = one batch)
        "dump_period": 10000,
        # do we use several cores?
        "parallel": False,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": False,
        # min/max of parameters
        "min": -5,
        "max": 5,
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2,
        # Things I added
        "add_random": 0,
        "n_niches": 1000,
        # How many total policies are evaluated?
        "evals": 200000,
    }


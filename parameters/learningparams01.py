
class LearnParams:

    param_idx = 1

    n_stat_runs = 5

    # Neural Network:
    hid = 30

    # Evolve nn:
    sigma = 0.5
    learning_rate = 0.1
    n_policies = 100
    # Determines if it evolves half of the species or evolves one third and creates a new set of policies for the other third
    thirds = False

    # Evo learner
    n_gen = 1000
    n_top_gen = 500


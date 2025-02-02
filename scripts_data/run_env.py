import numpy as np
from AIC.view import view


def run_env(env, policies, p, use_bh=False, vis=False):
    bh_space = [[] for _ in range(p.n_agents)]
    n_move_choice = 2
    for i in range(p.time_steps):
        state = env.state()
        if vis:
            g = env.G()
            view(env, i, g)
        actions = []
        for i, policy in enumerate(policies):
            action = policy(state[i]).detach().numpy()
            actions.append(action)

            if use_bh:
                bh_space[i].append(action_space(action, p, n_move_choice))

        env.action(actions)
    if vis:
        g = env.G()
        view(env, p.time_steps, g)
    if not use_bh:
        return env.G()
    else:
        return env.G(), calc_bh(bh_space, p.n_agents, p.n_bh, env.agents)


def action_space(act_vec, p, n_move):
    idx = np.argmax(act_vec[:-n_move])
    poi_type = int(np.floor(idx / p.n_sensors))
    return np.concatenate(([poi_type], act_vec[-n_move:]))
    # return [poi_type, sum(act_vec[-n_beh:])]


def calc_bh(bh_vec, n_agents, n_bh, agents):
    # If there are counterfactual agents, include in b
    bh = np.zeros((n_agents, n_bh))
    for ag_i in range(n_agents):
        ag_bhs = np.array(bh_vec[ag_i])
        agent = agents[ag_i]
        # Take the average of the two behavior vectors
        bh[ag_i] = np.mean(ag_bhs, axis=0)[1:]
    return bh


def calc_bh_OLD(bh_vec, n_poi_types, n_agents, n_bh, agents, cf_in_bh):
    # If there are counterfactual agents, include in b
    bh = np.zeros((n_agents, n_bh))
    for ag_i in range(n_agents):
        ag_bhs = np.array(bh_vec[ag_i])
        agent = agents[ag_i]
        # List of the start / stop indices of where each set of POI behaviors falls in the behavior array
        poi_bh_idxs = list(range(0, (n_poi_types * 3) + 1, 3))
        for poi in range(n_poi_types):
            # Get all behaviors associated with this POI type
            all_poi_bhs = ag_bhs[ag_bhs[:, 0] == poi]
            if all_poi_bhs.size > 0:
                # Mean of all behaviors for that POI type
                poi_bh = all_poi_bhs.mean(axis=0)[1:]
                # Average the behavior means -- can't just sum because we have to keep it between [0, 1]
                if cf_in_bh or n_bh == 2:
                    bh[ag_i][poi] = np.sum(poi_bh) / 3
                else:
                    # Set the three behaviors associated with this POI to the mean of the behaviors
                    bh[ag_i][poi_bh_idxs[poi]:poi_bh_idxs[poi + 1]] = poi_bh

        if cf_in_bh:
            bh[ag_i][-3:] = np.array([max(agent.min_dist), min(agent.max_dist), np.average(agent.avg_dist)])

    bh = np.nan_to_num(bh, np.nan)
    # bh = np.reshape(bh, (n_agents, n_poi_types * n_beh))
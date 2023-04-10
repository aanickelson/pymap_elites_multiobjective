import numpy as np


def run_env(env, policies, p, use_bh=False):
    bh_space = [[] for _ in range(p.n_agents)]
    n_behaviors = 3
    for i in range(p.time_steps):
        state = env.state()
        actions = []
        for i, policy in enumerate(policies):
            action = policy(state[i]).detach().numpy()
            actions.append(action)

            if use_bh:
                bh_space[i].append(action_space(action, p, n_behaviors))

        env.action(actions)

    if not use_bh:
        return env.G()
    else:
        return env.G(), calc_bh(bh_space, p.n_poi_types, p.n_agents, n_behaviors, env.agents, p.counter)


def action_space(act_vec, p, n_beh):
    idx = np.argmax(act_vec[:-n_beh])
    poi_type = int(np.floor(idx / p.n_sensors))
    return np.concatenate(([poi_type], act_vec[-n_beh:]))
    # return [poi_type, sum(act_vec[-n_beh:])]


def calc_bh(bh_vec, n_poi_types, n_agents, n_beh, agents, counter):
    # If there are counterfactual agents, include the agent behaviors
    if counter:
        bh = np.zeros((n_agents, n_poi_types + 3))
    else:
        bh = np.zeros((n_agents, n_poi_types))
    for ag in range(n_agents):
        ag_bhs = np.array(bh_vec[ag])
        agent = agents[ag]
        for poi in range(n_poi_types):
            try:
                all_poi_bhs = ag_bhs[ag_bhs[:, 0] == poi]
                if all_poi_bhs.size > 0:
                    poi_bh = all_poi_bhs.mean(axis=0)[1:]
                    bh[ag][poi] = np.sum(poi_bh)
            except RuntimeWarning:
                pass
        if counter:
           bh[ag][-3:] = np.array([agent.min_dist, agent.max_dist, (agent.avg_dist[0] / agent.avg_dist[1])])

    bh = np.nan_to_num(bh, np.nan)
    # bh = np.reshape(bh, (n_agents, n_poi_types * n_beh))
    return bh

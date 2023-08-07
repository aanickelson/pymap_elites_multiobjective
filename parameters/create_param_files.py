from create_param_strings import string_to_save as str_gen
from os import getcwd, path
from itertools import combinations, product
import numpy as np
from AIC.poi import tanh_poi, linear_poi



def circle(n, world, r, add_rand=False):
    """
    Create points on a circle around a centroid
    :param n: number of points
    :param world: size of the world
    :param r: radius of the circle
    :return: x,y of points on a circle
    """
    c = [world / 2, world / 2]
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_off = [0] * n
    y_off = [0] * n
    if add_rand:
        x_off = np.random.uniform(-1., 1., n)
        y_off = np.random.uniform(-1., 1., n)
    x = r * np.cos(t) + c[0] + x_off
    y = r * np.sin(t) + c[1] + y_off
    return np.c_[x, y]

def grid(n, world):
    # Divide world size by number of points we want
    step = (world - 2) / int(np.ceil(np.sqrt(n)))
    start = ((world / 2) % step)
    pos = np.array(list(product(np.arange(start, world, step), repeat=2)))
    ret_array = pos
    n_in_grid = len(pos)
    if n < n_in_grid:
        skip_every = int(np.ceil(n_in_grid / (n_in_grid - n)))
        keep_idxs = np.mod(np.arange(n_in_grid), skip_every) != 0
        ret_array = pos[keep_idxs]
    return ret_array[:n]

def corners(n, world, d):
    """
    Generate points in the four corners, d distance from the center
    :param n: number of points to generate
    :param world: size of the world
    :param d: distance points should be from center
    :return:
    """
    center = np.array([world/2, world/2])
    corner_vals = [center + [-d, -d], center + [-d, d], center + [d, -d], center + [d, d]]
    idxs = [0, 1, 2, 3] * int(np.ceil(n / 4))
    idxs = idxs[:n]
    pts = [np.ndarray.tolist(corner_vals[i] + np.random.uniform(-1, 1, size=2)) for i in idxs]
    return pts


def gen_all_files():
    # Values to iterate over
    n_cf = [0, 1, 5, 9]
    dist_to_center = [8]
    ag_in_st = [False, True]
    ag_in_bh = [False, True]
    # ag_in_st = [False, True]
    vals_to_iterate = [n_cf, dist_to_center, ag_in_st, ag_in_bh]
    combos = list(product(*vals_to_iterate))

    # Standard values
    world_size = 20
    poi_dist_to_center = 6
    n_poi = 16 # [10, 15, 20]
    # poi_locs = np.ndarray.tolist(grid(n_poi, world_size))
    pois = circle(n_poi, world_size, poi_dist_to_center, True)
    poi_locs = np.ndarray.tolist(pois)
    poi_class = (['linear_poi', 'tanh_poi'] * int(np.ceil(n_poi / 2)))[:n_poi]
    agent_loc = corners(1, world_size, 0)

    for combo in combos:
        start_num = 300
        cf, d, ag_st, cf_bh = combo
        add_val = 0
        if ag_st:
            add_val += 10
        if cf_bh:
            add_val += 100
        cf_locs = corners(cf, world_size, d)
        trial_num = start_num + add_val + cf
        generated_string = str_gen(trial_num, cf, cf_bh, world_size, True, True, cf_locs, poi_locs, n_poi, agent_loc, ag_st)
        filesave(generated_string, trial_num)

    return pois


def filesave(str_to_save, filenum):
    filename = "parameters{:03d}.py".format(filenum)
    filepath = path.join(getcwd(), filename)
    # Writing to file
    with open(filepath, "w") as fl:
        # Writing data to a file
        fl.writelines(str_to_save)



if __name__ == '__main__':
    poi_pos = gen_all_files()
    from matplotlib import pyplot as plt

    counter_locs = [[2.8602098625437753, 1.5772234705185684], [2.1540002623523646, 17.74362372813859],
                    [18.53467814194324, 1.918471564126539], [17.302498605261977, 17.550713507266],
                    [1.095394173470764, 2.727767777227816], [1.8924398652370835, 17.583733548290475],
                    [17.862221568582086, 1.6983992899553664], [17.7311752902783, 17.315501893932677],
                    [2.8378728517443466, 1.2468080813796043]]
    cf_pts = np.array(counter_locs)

    pts = np.array(poi_pos)
    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.scatter(pts[:, 0], pts[:, 1])
    plt.scatter(cf_pts[:, 0], cf_pts[:, 1], c='red')
    plt.show()
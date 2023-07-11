from create_param_strings import string_to_save as str_gen
from os import getcwd, path
from itertools import combinations, product
import numpy as np
from AIC.poi import tanh_poi, linear_poi



def circle(n, world, r):
    """
    Create points on a circle around a centroid
    :param n: number of points
    :param world: size of the world
    :param r: radius of the circle
    :return: x,y of points on a circle
    """
    c = [world / 2, world / 2]
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = r * np.cos(t) + c[0]
    y = r * np.sin(t) + c[1]
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
    move_poi = [[0, False, False], [0, False, False], [10, True, False], [110, True, True]]
    vals_to_iterate = [n_cf, dist_to_center, move_poi]
    combos = list(product(*vals_to_iterate))

    # Standard values
    world_size = 20
    n_poi = 16 # [10, 15, 20]
    poi_locs = np.ndarray.tolist(grid(n_poi, world_size))
    poi_class = (['linear_poi', 'tanh_poi'] * int(np.ceil(n_poi / 2)))[:n_poi]
    agent_loc = corners(1, world_size, 0)

    for combo in combos:
        start_num = 0
        cf, d, [add_val, move, poi] = combo
        cf_bh = False
        if cf:
            cf_bh = True
        cf_locs = corners(cf, world_size, d)
        trial_num = start_num + add_val + cf
        generated_string = str_gen(trial_num, cf, cf_bh, world_size, move, poi, cf_locs, poi_locs, n_poi, agent_loc)
        filesave(generated_string, trial_num)


def filesave(str_to_save, filenum):
    filename = "parameters{:03d}.py".format(filenum)
    filepath = path.join(getcwd(), filename)
    # Writing to file
    with open(filepath, "w") as fl:
        # Writing data to a file
        fl.writelines(str_to_save)


if __name__ == '__main__':
    gen_all_files()

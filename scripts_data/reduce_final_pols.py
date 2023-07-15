import numpy as np
from pymap_elites_multiobjective.scripts_data.plot_pareto import file_setup
import os
from pymap_elites_multiobjective.scripts_data.plot_2d_map import process_and_plot


def process(data):
    new_data = []
    for l in data:
        if sum(l[:2]) > 0.001:
            new_data.append(l)
    ret_data = np.array(new_data)
    return ret_data


def write_array(a, fl):
    for i in a:
        fl.write(str(i) + ' ')


def create_red_file(file_path, new_fpath):
    if os.path.exists(new_fpath):
        print(f"you've already done this one: {new_fpath}")
        return

    in_data = np.loadtxt(file_path)
    out_data = process(in_data)
    with open(new_fpath, 'w') as f:
        for ln in out_data:
            write_array(ln, f)
            f.write("\n")


if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     sys.exit('Usage: %s centroids_file archive.dat [min_fit] [max_fit]' % sys.argv[0])
    import os
    dates = ['516_20230711_150701']
    plot_exts = ['.png']
    n_niches = 5000
    n_pols = 200000
    bh_size = [5, 6, 9]

    graphs_f = os.path.join(file_setup(dates, 'bh_'), 'bh')
    for date in dates:
        root_dir = os.path.join(os.getcwd(), 'data', date)
        sub_dirs = list(os.walk(root_dir))[0][1]
        for d in sub_dirs:
            # if "129_run0" not in d:
            #     continue
            file_path = os.path.join(root_dir, d, f'archive_{n_pols}.dat')
            if not os.path.exists(file_path):
                print(f"Archive does not exist in {d}")
                continue

            new_fpath = os.path.join(root_dir, d, f'fin_arch_reduced.dat')
            graphs_f = os.path.join(file_setup(dates, 'bh_'), 'bh')
            if not os.path.exists(graphs_f):
                os.mkdir(graphs_f)

            it_worked = False
            for bh in bh_size:
                cent_f = os.path.join(root_dir, d, f'centroids_{n_niches}_{bh}.dat')
                if os.path.exists(cent_f):
                    it_worked = True
                    break

            if not it_worked:
                print(f"Could not find centroids file in {d}")
                continue


            create_red_file(file_path, new_fpath)
            f_info = (os.path.join(root_dir, d), cent_f, new_fpath, graphs_f)
            process_and_plot(f_info, [[0, 3]], plot_exts, d, red=True)


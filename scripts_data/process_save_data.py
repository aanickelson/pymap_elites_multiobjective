import numpy as np
import os
import re
import plot_2d_map_gym as plot_bh
import plot_pareto as plot_par
import csv


def get_files(names_dates, rootdir):
    all_files = []
    for dom, dates in names_dates:
        dom_dir = os.path.join(rootdir, dom)
        for date in dates:
            dt_num = re.split('_', date)[0]
            date_file = os.path.join(dom_dir, date)
            # This will give a list of [full path, [sub dirs], [sub files]] for each sub-folder in date_file
            # Skip the first, since that's the root of date_file
            sub_dirs = list(os.walk(date_file))[1:]
            for pth, _, fl in sub_dirs:
                all_files.append([pth, fl, dom, dt_num])

    return all_files


if __name__ == '__main__':
    root_dir = os.path.join(os.getcwd(), 'data')
    nm_dates = [['hopper', ['003_20240103_170210', '004_20240104_094025']],
                ['mountain', ['005_20231211_160926']],
                ['rover', ['dummy']]]
    file_info = get_files(nm_dates, root_dir)

    # Create a long file name of first letter of domain & domain numbers -- it's just a bunch of nested loops essentially
    fnamelist = ''.join([''.join([f'{nm[0]}_', ''.join([f'{d[:3]}_' for d in dt])]) for nm, dt in nm_dates])
    data_f_name = f'NOTES_fin_vals_{fnamelist}.csv'
    save_data_file = os.path.join(root_dir, 'sum data', data_f_name)

    n_gen = 100000
    n_obj = 2
    origin = np.array([0.] * n_obj)

    a = f'archive_{n_gen}.dat'

    all_data = []
    for fname, files, dom_nm, date_num in file_info:
        cent_files = [fn for fn in files if 'centroid' in fn]


        if (a not in files) or (not cent_files):
            continue

        cent = cent_files[0]
        bh_name = re.split('_|/', fname)[-2]
        run_num = int(fname[-1])

        # Get behavior and centroid numbers from centroid file name
        cent_split = re.split('_|\.', cent)
        n_bh = int(cent_split[-2])
        n_cent = int(cent_split[-3])

        # Load data
        arch_file = os.path.join(fname, a)
        cent_file = os.path.join(fname, cent)
        fits, desc = plot_bh.load_data(arch_file, n_bh, n_obj)

        # Pareto data
        is_eff = plot_par.is_pareto_efficient_simple(fits)
        hypervol = plot_par.get_area(fits[is_eff], origin)
        pol_kept, pol_par = len(is_eff), sum(is_eff)

        # Number of niches filled divided by total number of niches
        _, counts = np.unique(desc, return_counts=True, axis=0)
        bh_pct_full = len(counts) / n_cent

        run_info = [dom_nm, date_num, run_num, bh_name, hypervol, pol_par, pol_kept, bh_pct_full]
        all_data.append(run_info)

    # field names
    fields = ['dom name', 'date num', 'run num', 'bh name', 'hypervol', 'n pareto', 'n pol', 'bh pct full']

    with open(save_data_file, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(all_data)






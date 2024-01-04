from pymap_elites_multiobjective.scripts_data.plot_pareto import *

if __name__ == '__main__':

    # Change these parameters to run the script
    n_files = 10  # Need this in order to make sure the number of data points is consistent for the area plot

    ftypes = ['.svg', '.png']   # What file type(s) do you want for the plots

    plot_scatters = True   # Do you want to plot the scatter plots of the objective space for each data set

    # If you don't define this, it will use the current working directory of this file
    basedir_moo = "/home/anna/PycharmProjects/MOO_playground/"
    # 'lander_run0'
    dates_moo = ['006_20231128_114031']
    # basedir_qd = os.getcwd()
    # dates_qd = ['529_20230822_120257', '530_20230823_111127', '531_20230825_111600', '532_20230828_101445', '533_20230828_113856', '534_20230828_134557']

    files_info = [[dates_moo, basedir_moo, 'fits', '_n']]  # , [dates_qd, basedir_qd, 'archive_', '_q']]
    # FOR PARAMETER FILE NAME CODES -- see __NOTES.txt in the parameters directory

    # all_sets is a little wonky, I'll admit.
    # Each set is [[param file numbers], [param names for plot], 'graph title']
    # Param names provides the name of each parameter being compared. Should line up with the files
    # In this example, the names are consistent across all the plots, but they won't always be depending on what you want to run
    # param_names = ['MOME - No', 'C-MOME - Static', 'C-MOME - Move', 'C-MOME - Task', 'NSGA -No', 'NSGA - Static', 'NSGA - Move', 'NSGA - Task']
    # param_sets = ['100100_q', '100119_q','110119_q', '111119_q', "100100_n", "100109_n", "110109_n", "111109_n"]
    # nm = 'NSGA and QD by type'

    param_names = ['lander']
    param_sets = ['lander_n']
    nm = 'lander'


    # You shouldn't need to change anything beyond here
    # ---------------------------------------------------------

    graphs_fname = file_setup(dates_moo, basedir_moo)
    evols = [i * 100 for i in range(n_files)]
    # evols[-1] -=2
    data_and_nm = {p: [param_names[i]] for i, p in enumerate(param_sets)}
    plot_fname = f'{nm}'  # What domain is being tested

    for dates, basedir, arch_or_fits, nm_append in files_info:
        files = get_file_info(dates, arch_or_fits, basedir)

        # Walk through all the files in the given directory
        for sub, date, params_name, fnums in files:
            # Pulls the parameter file number
            # p_num = params_name[:3]
            p_num = params_name.split('_')[0] + nm_append
            if not p_num in param_sets:
                # print(f'Did not save data for {params_name} in {sub}')
                continue
            if len(fnums) < n_files:
                continue

            # This block goes through each fil609047338827017e, gets the data, finds the pareto front, gets the area, then saves the area
            areas, x_p, y_p = get_areas_in_sub(sub, fnums, p_num, plot_scatters, date, params_name, graphs_fname, ftypes, arch_or_fits, origin=[-150, -150])[:n_files]
            if len(areas) < n_files:
                continue

            print(f'{p_num}, {areas[-1]}')
            try:
                data_and_nm[p_num].append(areas)
            except KeyError:
                print(f'Did not save data for {params_name} in {sub}')
                continue

    # Plot the areas data for all parameters on one plot to compare
    plot_areas(evols, data_and_nm, plot_fname, graphs_fname, ftypes)

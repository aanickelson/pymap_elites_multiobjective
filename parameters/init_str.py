from os import getcwd, path


def line_str(num):
    return f"from pymap_elites_multiobjective.parameters.parameters{num:03d} import Parameters as p{num:03d}\n"

def init_strings():
    s = ""
    ranges = [[10, 10, 1], [11, 13, 1], [21, 23, 1], [31, 33, 1], [41, 43, 1],
              [121, 123, 1], [141, 143, 1], [231, 239, 2], [241, 249, 2], [341, 349, 2]]
    rng_names = ['no_cf', 'no_close', 'move_close', 'no_far', 'move_far',
                 'poi_close', 'poi_far', 'no_far_new', 'move_far_new', 'poi_far_new']
    batches = ''
    for idx, [lw, hg, by] in enumerate(ranges):
        batch_nm = f'{rng_names[idx]} = ['
        for i in range(lw, hg+1, by):
            s += line_str(i)
            batch_nm += f'p{i:03d},'
        batch_nm += ']\n'
        batches += batch_nm
    s += batches
    return s


def filesave(str_to_save):
    filename = "__init__.py"
    filepath = path.join(getcwd(), filename)
    # Writing to file
    with open(filepath, "w") as fl:
        # Writing data to a file
        fl.writelines(str_to_save)


if __name__ == '__main__':
    string_to_save = init_strings()
    filesave(string_to_save)

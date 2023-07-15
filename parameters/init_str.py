from os import getcwd, path


def line_str(num):
    return f"from pymap_elites_multiobjective.parameters.parameters{num:03d} import Parameters as p{num:03d}\n"

def init_strings():
    s = ""
    pnums = [0, 1, 2, 3, 5, 9,
             11, 12, 13, 15, 19,
             111, 112, 113, 115, 119,
             129, 500]
    for i in pnums:
        s += line_str(i)
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

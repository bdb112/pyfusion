import runpy as sys_runpy


def runpy(cmdline):
    """ Allows scripts to be run form the command line, compatible with both pythons.
    Respects process_cmd_line_args, but so far, you have to be more care with
    quotes, and locals/globals are not accessible

    for cpt in 'Axial,Outward,Upward'.split(','):
        runpy("pyfusion/examples/plot_correlation.py diag1='H1Toroidal{}' diag2='ElectronDensity_1' bp=[20e3,2e3] shot_number=100618 hold=1".format(cpt))

    """
    path, rest = cmdline.split(' ', 1)
    print(path)
    # rest = ','.join(["'" + tok + "'" if "'" not in tok and tok not in locals() else tok for tok in rest.split(' ')])
    rest_commas = ','.join(rest.split(' '))
    print(rest)
    exec('cmdline_vars = dict(' + rest_commas + ')')
    init_globals = dict(cmdline_vars=cmdline_vars, from_runpy='foo', args_from_runpy=rest)
    ret = sys_runpy.run_path(path, init_globals=init_globals)

import runpy as sys_runpy
import pyfusion
import time as tm
""" Seems to be python 3 compatible """

def runpy(cmdline, vardict=None):
    """ Allows scripts to be run from the command line, compatible with both pythons.
    Respects process_cmd_line_args, but so far, you have to be more careful with
    quotes, and locals/globals are not accessible by default - each run is in a 
    new space, although the same figure can be accessed.
    
    # the original version - simple, can do loops, but harder to extract the results of multiple runs
    for cpt in 'Axial,Outward,Upward'.split(','):
        runpy("pyfusion/examples/plot_correlation.py diag1='H1Toroidal{}' diag2='ElectronDensity_1' bp=[20e3,2e3] shot_number=100618 hold=1".format(cpt))

    # better, more general version - format is replaced by a second argument, vardict which has all the necessary variables
    # this also allows multidimensional scans, storing data in a global list variable (e.g. pyfusion.GL)

        for shot in range(100626,100635):  runpy("pyfusion/examples/plot_correlation.py diag1='H1ToroidalAxial' diag2='ElectronDensityAll' bp=[20e3,20e3] shot_number={s} t_range=[0.03,.1] mask1=[0,-1] mindf=1e6 coefft=0",vardict=dict(s=shot))

    # A one-liner for plotting is  (note that the label is not automatically generated)
    plot(*(transpose([[d['shot'], d['diag2']] for d in pyfusion.GL if 'cmd' in d and d['cmd']==pyfusion.GL[-1]['cmd']])),label='100626-634, 10/20')

    """
    orig_cmdline = cmdline
    if vardict is not None:
        cmdline = orig_cmdline.format(**vardict)  # make the substitutions
    path, rest = cmdline.split(' ', 1)
    print(path)
    # rest = ','.join(["'" + tok + "'" if "'" not in tok and tok not in locals() else tok for tok in rest.split(' ')])
    rest_commas = ','.join(rest.split(' '))
    print(rest)
    exec('cmdline_vars = dict(' + rest_commas + ')')
    GL = pyfusion.GL
    init_globals = dict(cmdline_vars=cmdline_vars, from_runpy='foo', args_from_runpy=rest, GL=pyfusion.GL)
    ret = sys_runpy.run_path(path, init_globals=init_globals)
    if isinstance(GL[-1], dict) and 'stamp' not in list(GL[-1]):
        GL[-1].update(dict(stamp = tm.strftime('%Y%m%d%H%M%S'), cmd=orig_cmdline, vardict=vardict))
    else:
        print('seems like there was no new list entry to stamp')

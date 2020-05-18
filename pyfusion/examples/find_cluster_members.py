""" Given a set of lists of angles, find those members closest to other members
  simple minded algorithm is slow and relies on choice of cutoff
Also contains some handy routines that should be somewhere in a library. 

  Simple minded but slow (O(num^2)) algorithm is:
  1/ Start with a 'centre' list (ctrs), which may be given, or defaults to the mean f all the lists mapped to a certain rage of 2pi
  2/ successively delete the furthest from the current 'centre' and recalculate the 'centre' until the end criterion is met:
   Either the maximum mean error meets a criterion or the a certain fraction of the members remain
   Not clear how to choose the 'centre' using means_
  version 1: centre evolves - get a lower spread, but mainly because the centre is optimised

A true clusterer would be better
"""
import numpy as np
from pyfusion.utils.utils import fix2pi_skips, modtwopi
from pyfusion.data.convenience import between, bw, btw, decimate, his, broaden

def dist_wrap(l1, l2, norm_index=2):
    """ return the average distance modulo 2pi between two lists of angles
    The average is calculated as an n-norm.
    >>> round(dist_wrap([2, 1, 0], [-4, 1, 3]), 7)
    3.013336

    """
    diffs = modtwopi(np.array(l1) - np.array(l2), offset=0)
    return np.linalg.norm(diffs, norm_index)

def get_probe_angles(input_data, closed=False):
    """  
    return a list of thetas for a given signal (timeseries) or a string that specifies it.
              get_probe_angles('W7X:W7X_MIRNOV_41_BEST_LOOP:(20180912,43)')

    This is a kludgey way to read coordinates.  Should be through acquisition.base or
    acquisition.'device' rather than looking up config directly
    """
    import pyfusion
    if isinstance(input_data, str):
        pieces = input_data.split(':')
        if len(pieces) == 3:
            dev_name, diag_name, shotstr = pieces
            shot_number = eval(shotstr)
            dev = pyfusion.getDevice(dev_name)
            data = dev.acq.getdata(shot_number,diag_name, time_range=[0,0.1])
        else:
            from pyfusion.data.timeseries import TimeseriesData, Timebase, Signal
            from pyfusion.data.base import Channel, ChannelList, Coords
            input_data = TimeseriesData(Timebase([0,1]),Signal([0,1]))
            dev_name, diag_name = pieces
            # channels are amongst options
            opts = pyfusion.config.pf_options('Diagnostic', diag_name)
            chans = [pyfusion.config.pf_get('Diagnostic', diag_name, opt)
                     for opt in opts if 'channel_' in opt]
            # for now, assume config_name is some as name
            input_data.channels = ChannelList(*[Channel(ch, Coords('?',[0,0,0])) for ch in chans])

    Phi = np.array([2*np.pi/360*float(pyfusion.config.get
                                      ('Diagnostic:{cn}'.
                                       format(cn=c.config_name if c.config_name !='' else c.name), 
                                       'Coords_reduced')
                                      .split(',')[0]) 
                    for c in input_data.channels])

    Theta = np.array([2*np.pi/360*float(pyfusion.config.get
                                        ('Diagnostic:{cn}'.
                                       format(cn=c.config_name if c.config_name !='' else c.name),
                                         'Coords_reduced')
                                        .split(',')[1]) 
                      for c in input_data.channels])

    if closed: 
        Phi = np.append(Phi, Phi[0])
        Theta = np.append(Theta, Theta[0])
    return(dict(Theta=Theta, Phi=Phi))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    _var_defaults = """
min_left = 30
tolerance= 0.2
# DAfile = 'DAMIRNOV_41_13_15_3m_20180808_5.npz'
# DAfile = ''
# 'DAMIRNOV_41_13_BEST_LOOP_10_3ms_20180912043.npz'
# 'W7X_MIR/DAMIRNOV_41_BEST_LOOP_REST_15_3m_2018fl.npz'
DAfile = 'W7X_MIR/DAMIRNOV_41_13_nocache_15_3m_2018fl.npz'
version=0  # 0 fixed ctrs, 1 evolves
"""
    exec(_var_defaults)
    #exec(';'.join(_var_defaults.split()))

    from pyfusion.utils import process_cmd_line_args
    exec(process_cmd_line_args())

    # Slow: around 1000 13 element lists/second - probably could eliminate
    #  more than one at a time.  However it may make more sense to look for
    #  clusters - this will work if there are two types of shapes.

    from pyfusion.data.DA_datamining import DA, report_mem, append_to_DA_file


    if DAfile != '':
        da = DA(DAfile, load=1)
    else:
        print('try for a da in locals()')
        try:
            da
        except NameError as reason:
            print(' no da - did you use run -i?')
    wh = da['indx']     # all of them
    #wh = np.where((btw(da['freq'],2,4)) & (da['amp']>0.012) )[0]
    wh = np.where(da['shot'] == 180904035)[0]
    phs = da['phases'][wh].tolist()
    inds = da['indx'][wh].tolist()

    ctrs = np.mean(modtwopi(phs, 0), 0)
    num = len(ctrs)

    print(len(wh))

    for left in range(len(phs), min_left, -1):
        if version == 1:
            ctrs = np.mean(modtwopi(phs, 0), 0)
        dists = [dist_wrap(ctrs, phases)/num for phases in phs]
        if np.max(dists) < tolerance:
            break
        worst = np.argmax(dists)
        phs.pop(worst)
        inds.pop(worst)
        """
        worst = np.argsort(dists)[::-1]  # this won't work as position in list changes....
        for wind in range(int(len(worst) * 0.01)):
            phs.pop(wind)
            inds.pop(wind)
        """
    print('{num} found, average scatter is {avg:.2f} rad'.format(num=len(inds), avg=np.average(dists)))

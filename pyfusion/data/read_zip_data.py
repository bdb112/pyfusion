import zipfile
import sys
import numpy as np


def read_zip_data(filename, trans={}, verbose=0):
    """ General code to read a zipped bunch of text files named according to their variable
        Optional trans is a dictionary to translate the original name into a new name
        Assumes the shot number is the last part of the name before the discarded extension.
    """
    zdata = zipfile.ZipFile(filename)
    #  This until =======  is MPM-specific
    dat = dict(filename=filename,
               shot = divmod(int(2e10)+int(zdata.namelist()[0].split('.')[0].split('_')[-1]), 1000))
    for nm, finfo in zip(zdata.namelist(), zdata.filelist):
        # get first token or first two if error
        nmroot = nm.split('.')[0]  # get rid of .txt or .dat
        # allow for names like Te_error (OK, not too specific) and _LP (very MPM specific)
        temp_name = '_'.join(nmroot.split('_')
                             [0:1 + np.any([frag in nmroot for
                                            frag in '_LP,_vs,_error'.split(',')])])
        my_name = trans[temp_name] if temp_name in trans else temp_name
        # could use loadtxt here, but might need to convert to a stream
        # advantage would be to seamlessly read multi columns
        datarr = np.array([float(ln) for ln in zdata.read(finfo).split()])
        print(len(datarr), my_name, temp_name, finfo)
        dat.update({my_name: datarr})
    print(dat['shot'], dat.keys())
    # End of MPM specific stuff
    if verbose > 0:
        print('no checks yet') # temp_name is not a list
        # unchanged = [t_name for t_name in temp_name if t_name in dat.keys()]
        # if len(unchanged) > 0:
        #    print('no translation of ', unchanged)

    return(dat)

def read_MPM_data(filename='/data/databases/W7X/MPM/MPM_20160309_13.zip', verbose=1,
                  trans=dict([('te','Te'),('te_error', 'eTe'),('ne','ne18'),('ne_error','ene'),
                              ('T','t_mid'),('shot','old_shot')])):
    """ read a bunch of ascii files from the MPM diagnostic (Philipp Drews)
    The trans dictionary changes to my naming convention.
    """
    dat = read_zip_data(filename=filename, trans=trans, verbose=verbose)
    shp = np.shape(np.array([dat[k] for k in ['x','y','z']]))
    if len(shp) != 2:
        print('************************ x, y and z not all the same length ****************')
        
    if verbose > 0:
        R = np.linalg.norm([dat['x'], dat['y']], axis=0)
        print('shot {shot}, {filename:s}, x0={x0:.4f}'
              .format(filename=filename, shot=dat['shot'], x0 = dat['x'][0]))
        print('time range={t0:.4f}:{t1:.4f}, R_range={r0:.4f}:{r1:.4f}, dr={dr:.4f}'
              .format(t0=dat['t_mid'][0], t1=dat['t_mid'][-1], 
                      r0=R[0], r1=np.min(R), dr=R[0] - np.min(R),))


    return(dat)


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else '/data/databases/W7X/MPM/MPM_20160309_32.zip'
    read_zip_data(filename)

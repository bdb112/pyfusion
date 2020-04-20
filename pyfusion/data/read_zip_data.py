import zipfile
import sys
import numpy as np


def read_zip_data(filename, trans={}, verbose=0):
    """ General code to read a zipped bunch of text files named according to their variable
        Optional trans is a dictionary to translate the original name into a new name
        Assumes the shot number is the last part of the name before the discarded extension.
    """
    zdata = zipfile.ZipFile(filename)
    dat = dict(filename=filename,
               shot = divmod(int(2e10)+int(zdata.namelist()[0].split('.')[0].split('_')[-1]), 1000))
    for nm, finfo in zip(zdata.namelist(), zdata.filelist):
        # get first token or first two if error
        nmroot = nm.split('.')[0]  # get rid of .txt or .dat
        temp_name = '_'.join(nmroot.split('_')[0:1 + int('error' in nmroot)])
        my_name = trans[temp_name] if temp_name in trans else temp_name
        dat.update({my_name: np.array([float(ln) for ln in zdata.read(finfo).split()])})
    print(dat['shot'], dat.keys())
    if verbose > 0:
        print('no checks yet') # temp_name is not a list
        # unchanged = [t_name for t_name in temp_name if t_name in dat.keys()]
        # if len(unchanged) > 0:
        #    print('no translation of ', unchanged)

    return(dat)

def read_MPM_data(filename='/data/databases/W7X/MPM/MPM_20160309_13.zip', verbose=1,
                  trans=dict([('te','Te'),('te_error', 'eTe'),('ne','ne18'),('ne_error','ene'),('T','t_mid')])):
    """ read a bunch of ascii files from the MPM diagnostic (Philipp Drews)
    """
    dat = read_zip_data(filename=filename, trans=trans, verbose=verbose)
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

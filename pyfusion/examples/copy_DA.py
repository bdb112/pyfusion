""" timing on E4300  

72 sec translate PF2_130813_50_5X_1.5_5b_rms_1_diags.npz' to 2.6G blosc 2
162  zlib 2 1.4GB
300 zlib 5  

              zlib 2   zlib3     zlib5      blosc1   blosc2              npz
phases       19.5MB/s  15MB/sec  12.5MS    356MB/sec   
               35%      35%      36.8        0%                    37%
freq           58%                60%     
phase deg/10  *48MB/s                               *620M
               *68%                                 *50%

in 8GB system, took 528 sec to read 7.9GB phase from 8X
* means relative to 4bye float.
"""
import tables as tb
import os
import numpy as np
from bdb_utils import process_cmd_line_args
from time import time as seconds
from pyfusion.data.DA_datamining import DA, report_mem


_var_defaults = """
DAfilename='/data/datamining/PF2_130813_50_5X_1.5_5b_rms_1_diags.npz'
outfilename=None
keep_open = 0
complevel=2
complib = 'zlib'
var='phases'
"""
exec(_var_defaults)
exec(process_cmd_line_args())

filters=tb.Filters(complevel=complevel, complib=complib)

dd = DA(DAfilename).da

if outfilename is None:
    (base, ext) = os.path.splitext(os.path.realpath(DAfilename))
    outfilename = base + os.path.extsep + 'h5'

outf = tb.openFile(outfilename, "a")

for var in dd.keys():

    st_copy = seconds()
    if var in [nd.name for nd in outf.listNodes('/')]:
        raise LookupError('{f} already has a node "{n}"'
                          .format(f=outf.filename, n=var))
    val = dd[var]  # need to hold it in memory this way to avoid multiple access
    sizebytes = val.nbytes
    print('{dt:.1f}s to read {v} {GB:.2f} GB for {f}'
          .format(dt=seconds()-st_copy, 
                  GB = sizebytes/1e9, f=os.path.split(outfilename)[-1], v=var))

    st_write = seconds()

    try:
        var_atom = tb.atom.Atom.from_dtype(numpy.dtype(val.dtype))
    except Exception, reason:
        print('failed to copy {v}, reason: {r}'
              .format(v=var, r=reason))
        continue
    result = outf.createCArray(outf.root, var, atom=var_atom, shape=val.shape, filters=filters)
    result[:] = val

    dt_copy = seconds() - st_copy
    dt_write = seconds() - st_write

    st_slice = seconds()
    res0 =  result[len(shape(result))*(0,)]
    dt_slice = seconds() - st_slice
    MB = np.product(result.shape)*res0.nbytes/1e6

    try:
        filesize = os.stat(outf.filename)[6]
    except:
        filesize = np.nan
        print('error obtaining filename....')

    print('{dt_write:.2f} to write, '
          '{MBsec:.1f} MB/sec, compr={compr:.1f}%'
          .format(dt_write = dt_write,
            dt_copy = dt_copy, MBsec = MB/dt_write,
            compr = 100*(1-filesize/MB/1e6)))

if (not keep_open):
    outf.close()

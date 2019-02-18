"""

Simple save to json:  json.dump(mdat, file('LP/LP20160224_25_L57.json', 'wt'))
"""

import numpy as np
import os, sys
import json
import pyfusion

sep = ','
nl = '\n'
ofilename = None

LPfile = 'LP/20160224_25_L53.npz'
if len(sys.argv) > 1 and sys.argv[1][0] != '-':
    LPfile = sys.argv[1]

if ofilename is None:
    path, nameext = os.path.split(LPfile)
    (name, ext) = os.path.splitext(nameext)
    ofilename = os.path.join(path, name + '.csv')
    jfilename = os.path.join(path, name + '.json')

try:
    from pyfusion.data.DA_datamining import DA
    da = DA(LPfile)
    dat = da.da
    masked = da.masked

except ImportError:
    raise
    datnpz = np.load(LPfile)
    dat = {}
    for k in datnpz:
        dat[k] = datnpz[k].tolist()
    masked = dat

info = dat.get('info')
params = info['params']

channels = info['channels']
if channels[0][0] == '_':
    channels = ['W7X'+chan+'_I' for chan in channels]

short_channels = []
for chan in channels:
    seg = chan.split('L5')[1][0] 
    if  seg =='3':
        suffix = 'U' # upper limiter.
    elif seg =='7':
        suffix = 'L'
    else:
        raise ValueError('unknown limiter segment number {s}'.format(s=seg))

    short_channels.append(chan.split('LP')[1].split('_')[0]+suffix)
"""
First line: version, channels, samples[1], header line[2], date, progId, itc in ns
Second line: (values of above items)
Third line: json data for LP file
4th and onwards (channels in total): json data for each channel (initially only one?)
next line: Headers e.g. shot, date, progId, t_mid, Te, I0, ne18, nits, resid, mask
Each channel is labelled according to the probe number  e.g. Te_20U means upper limiter, probe 20

[1] the number of time samples.
[2] the line which contains the data column headers, starting from 1.  The data immendiately follows.  This allows skipping the json data.

"""

samples = len(dat['t_mid'])
nchans = len(channels)

ofile = open(ofilename,'wt')
ofile.write("'version','channels','samples','header line','date','progId','utc_ns','dtseg'\n"\
            "{version},{channels},{samples},{hl},{date},{progId},{utc_ns},{dtseg}\n".replace(',',sep)
            .format(channels=nchans, version=1, samples=samples,
                    s=sep, hl=2 + 2 + nchans,date=dat['date'][0],progId=dat['progId'][0],
                    utc_ns = params['i_diag_utc'][0],dtseg=params['dtseg']))
json.dump(info, ofile)
ofile.write(nl)

for ch in channels:
    try:
        # if these are local, their info may correspond more closely to that used
        dev_name = 'W7X'
        shot_number = [dat['date'][0], dat['progId'][0]]
        dev = pyfusion.getDevice(dev_name)
        cdata = dev.acq.getdata(shot_number, 'W7X_'+ch+'_I')
        opts = cdata.params
    except:
        opts = pyfusion.conf.utils.get_config_as_dict('Diagnostic', ch)
        for k in list(opts):
            if k not in ['gain', 'area', 'params', 'coords_w7_x_koord']:
                opts.pop(k)
                
    # no need for this huge array in the DA file
    [opts.pop(k) for k in list(opts) if k in ['diff_dimraw']]
    opts['name'] = ch
    json.dump(opts, ofile)
    ofile.write('\n')

vars = list(dat)
vars.sort()
vars.remove('info')
svars = ['t_mid','shot','date','progId','indx']
for v in svars:
    vars.remove(v)

if 't' in da:
    svars.insert(0,'t')

for k in ['ne18']:
    vars.remove(k)
    vars.insert(3, k)

#ofile.write(sep.join(["'{k}'".format(k=k) for k in svars + vars]) + nl)

ofile.write(sep.join(["'{k}'".format(k=k) for k in svars]))
for c in range(nchans):
    ofile.write(sep + sep.join(["'{k}_{c}'".format(k=k, c=short_channels[c]) for k in vars]))
ofile.write(nl)

mdat = {}  #make a new dict with masked variables where they exist and ordinary if not.
for k in vars + svars:
    if k in  dat['info']['valid_keys']:
        md = masked[k]
    else:
        md = da[k]
    # some matlab doesn't recognize inf?  loadjson.m (mathworks, qianqian fang 2011/09/09 seems to want to
    winf = np.where(np.isinf(md))
    if len(winf[0]) > 0:
        print('replacing {n} "infinity" values per channel with np.nans'
              .format(n=np.product([len(dim) for dim in winf])))
    
        md[winf] = np.nan
    mdat[k] = md.tolist()

for s in range(samples):
    ofile.write(sep.join(['{val}'.format(val=mdat[k][s]) for k in svars]))
    for c in range(nchans):
        ofile.write(sep+sep.join(['{val}'.format(val=mdat[k][s][c]) for k in vars]))
    ofile.write(nl)

print('\nWrote ' + ofile.name)
ofile.close()
# now the json file
mdat.update(dict(info=dat['info']))
import json
json.dump(mdat, file(jfilename, 'wt'))
print('Wrote ' + jfilename)

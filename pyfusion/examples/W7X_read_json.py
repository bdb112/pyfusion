#find /data -iname *json*|grep from|grep -v Samples

files = """
/data/databases/W7X/cache/archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.82/DataModuleDesc.182_DATASTREAM/1/Channel_1/scaled/_signal.json?from=1456930821345103981&upto=1456930888345103980
/data/databases/W7X/cache/archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.82/DataModuleDesc.182_DATASTREAM/3/Channel_3/scaled/_signal.json?from=1457015819400000000&upto=1457015821000000000
/data/databases/W7X/cache/archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.82/DataModuleDesc.183_DATASTREAM/4/Channel_4/scaled/_signal.json?from=1456932404467103981&upto=1456932471467103980
/data/databases/W7X/cache/archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.82/DataModuleDesc.190_DATASTREAM/0/Channel_0/_signal.json?from=1455802229271153060&upto=1455802230268617061
/data/databases/W7X/cache/archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.82/DataModuleDesc.190_DATASTREAM/2/Channel_2/scaled/_signal.json?from=1456932404467103981&upto=1456932471467103980
/data/databases/W7X/cache/archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.82/DataModuleDesc.190_DATASTREAM/5/Channel_5/scaled/_signal.json?from=1456930821345103981&upto=1456930888345103980
/home/bdb112/pyfusion/working/pyfusion/archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.82/DataModuleDesc.182_DATASTREAM/1/Channel_1/scaled/_signal.json?from=1456930821345103981&upto=1456930888345103980
/data/databases/W7X/_signal.json@from=1457536066136171000&upto=1457536066635171001
/data/databases/W7X/webapi/archive-webapi.ipp-hgw.mpg.de/programs.json@from=1450137600000000000&upto=1450223999999999999
"""

import json
from numpy import sort, unique, diff
from numpy import savez_compressed as savez
from tempfile import TemporaryFile

files = files.split()

dicts = []
for fil in files:
    dic = json.load(open(fil))
    dicts.append(dic)
    if 'dimensions' in dic:
        dim = dic['dimensions']
        # dim += np.random.randint(0,10000000000000,[len(dim)])  # just to check - should be 6-8x longer than dim
        if len(unique(diff(dim))) > 1:
            print('{fil} has at least {n:,}/{t:,} anomalies'
                  .format(fil=fil, n=len(unique(diff(dim))), t=len(dim)))
        else:
            print('{fil} is clean!!'.format(fil=fil))
        outfile = TemporaryFile()
        savez(outfile, dim=diff(dim))
        print('compressed file length = {l:,}/{raw:,} bytes'.format(l=outfile.tell(),raw=8*len(dim)))

"""
#pyfusion.config.set('global','localdatapath','local_data') 

# this request translates to a json file which is stored locally - see below for complete example
xx=dev.acq.getdata([20160302,23],'W7X_L53_LP10_I',no_cache=1)
http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.82/DataModuleDesc.190_DATASTREAM/5/Channel_5/scaled/_signal.json?from=1457536002136103981&upto=1457536069136103980

# complete example - assuming you have a cache under the working directory e.g.
# /home/bdb112/pyfusion/working/pyfusion/archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.82/DataModuleDesc.190_DATASTREAM/5/Channel_5/scaled/_signal.json?from=1456930821345103981&upto=1456930888345103980
#
import pyfusion
pyfusion.LAST_DNS_TEST=-1
pyfusion.CACHE=1
run pyfusion/examples/save_to_local.py "diag_name=['W7X_L53_LP10_I']" shot_list=[[20160302,23]]

# the diff_rawdim only takes up 19k! (because of the diff) (but python3 takes 28945!!)
 !unzip -v /tmp/20160302_23_W7X_L53_LP10_I.npz
 6200080  Defl:N  2143103  65% 2016-12-08 13:28 e737eb58  rawtimebase.npy
 3100080  Defl:N  1583864  49% 2016-12-08 13:28 7c85b5a6  rawsignal.npy
12400613  Defl:N    19644 100% 2016-12-08 13:28 147443c3  params.npy

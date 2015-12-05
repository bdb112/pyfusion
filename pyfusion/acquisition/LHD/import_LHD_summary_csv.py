""" read the LHD summary csv file and put in in a dictionary of arrays
where the index is the shot number.  This may require adding a "0" shot 
(apparently not as of Feb 2013.
Where possible, integers and reals
are converted, and the strings are reduced to the minimum length. (Note - this 
will cause errors if longer strings are added afterwards.
"""

from pyfusion.utils import read_csv_data
import numpy as np
import re

hack_merge_another_file=False

print('reading..')
lhd=read_csv_data.read_csv_data('LHD_Summary_Long.csv',header=3)
print('{k} keys, {n} entries read'.format(n=len(lhd['nShotnumber']), 
                                          k=len(lhd.keys())))

# this is hacked in because I missed GAMMA and another in my big file
if hack_merge_another_file:
    lhd2 = read_csv_data.read_csv_data('/home/bdb112/datamining/lhd_summary_data.csv',header=3)
    ksh='nShotnumber'
    ws2 = np.where(lhd[ksh] != lhd2[ksh])[0]
    if len(ws2) != 0: raise LookupError('{n} mismatched shots'.format(n=len(ws2)))
    # if we already have the key, give this one a different name -otherwise same
    for k in lhd2.keys(): 
        if k in lhd.keys(): lhd[k+'1']=lhd2[k]
        else: lhd[k]=lhd2[k]

def nansort(arr):
    return(np.sort(arr[np.where(np.invert(isnan(arr)))]))

def nanargsort(arr):
    return(np.argsort(arr[np.where(np.invert(isnan(arr)))]))


if __name__ == "__main__":

    """ Do it simply, not necessarily efficiently
    (after wasting 3 hours doing it efficiently)
    First delete all records with blank shot numbers by copying to tmp
    Then convert shot to int, and reorder everything to shot order
    Then create the final shot array, indexed by shot (must be equal or bigger len)
    The target address in the final array is just the shot coulum (sht) in the tmp
    Then for each column, find non blanks (wnn) 
    Prepare a target arrlen array of the right type, with nan entries (or -1, '')
    depost them target[sht[wnn]] = col[wnn]
    Finally, the shot column in the final array (Shot) should be == arange(maxshot+1) 

    """
    LHD = {}
    tmp = {}
    sh = 90091
    err=0
    str_summary=[]

    wnotnull = np.where(lhd['nShotnumber'] != '')[0]  # cautiously convert to int
    shots_tmp = lhd['nShotnumber'][wnotnull].astype(np.int32)
    # need unique here, are there are 2 shot 100's ! (what does this mean?)
    shots_test,ws = np.unique(shots_tmp, return_index=1)
    # reorder the strings in a new dict, in shot number order.
    for k in lhd.keys(): tmp.update({k: lhd[k][ws]})

    # now prepare the final shot array
    arrlen = np.max(shots_tmp)+1  # need a spot for all shots including 0
    shots = np.zeros(arrlen, dtype=np.int32) -1  # initialise to shot=-1
    shots[shots_tmp] = shots_tmp
    LHD.update({'Shot': shots})

    for k in tmp.keys():
        as_str_in_order = tmp[k]
        # now look for '' in other cols
        wcolnotnull = np.where(as_str_in_order != '')[0]

        chk_range = min(10, len(wcolnotnull))
        # get a lot of values, in case the first choice is not representative
        values = '_'.join([as_str_in_order[wcolnotnull[i]].strip() 
                            for i in range(chk_range)])
        if re.match('^[_0-9]*$',values): 
            dt = 'int32'
            arr = -np.ones(arrlen).astype(dt)
            wdecimal = np.where(
                np.remainder(as_str_in_order[wcolnotnull].astype(float),1)!=0)[0]
            if len(wdecimal)>0: 
                print('reverting {k} to float based on {eg}'
                      .format(k=k, eg=as_str_in_order[wcolnotnull[wdecimal[0]]]))
            dt = 'float32'
            arr = np.nan + np.ones(arrlen).astype(dt)

        elif re.match('^[_+-.0-9eE]*$',values): 
            dt = 'float32'
            arr = np.nan + np.ones(arrlen).astype(dt)
        else: 
            dt == 'str'
            #arr = np.empty(arrlen,dtype='|S256')  # need to initialise empty
            arr = np.array(arrlen*[''],dtype='|S256')


        try:  # the conversion may go wrong - protect
            arr[shots_tmp[ws[wcolnotnull]]] = \
                as_str_in_order[wcolnotnull].astype(np.dtype(dt))
        except Exception as details:
            err += 1
            print('Failed on {k} (type was based on "{v}" for shot {sh}, {d} {a}'
                  .format(k=k, d=details, a=details.args, v = values, sh=sh))

            arr = np.array(arrlen*[''],dtype='|S256')
            #arr = np.empty(arrlen,dtype='|S256')
            #arr = np.array(arrlen*[''])
            arr[shots_tmp[ws[wcolnotnull]]] = as_str_in_order[wcolnotnull]

            # compress, but beware assignments in the future.
            arr=np.array([s.strip() for s in arr])
            str_summary.append('{k}: {oldty}-> {dty}'
                               .format(k=k, dty=arr.dtype, 
                                       oldty=as_str_in_order.dtype))
            print('revert {k} to a string, type {dty}'.format(k=k, dty=arr.dtype))


        LHD.update({k: arr})  # add the new entry

    print('{err} string reversions/compressions'.format(err=err))
    print('{s}'.format(s=str_summary))

    for k in lhd.keys():
        if len(LHD[k]) != arrlen: print('conversion error on {k}'
                                         .format(k=k))

    wset = np.where(LHD['Shot'] != -1)[0]
    werr = np.where(LHD['Shot'][wset] != np.arange(arrlen)[wset])[0]
    if len(werr) > 0: raise LookupError('shot numbers mixed up')


    if 'y' in raw_input('save ? ').lower():
        fn = 'LHD_summary_new'
        print('saving as {n}..'.format(n=fn))
        np.savez_compressed(fn,LHD=LHD)
    else:
        print('not saved')


"""

reverting time_nlmax to float based on 0.511
reverting ECHToshiba3Power to float based on 0.132
reverting time_ipmax to float based on 0.486
reverting NBI2Power to float based on 586.68
Failed on ECHCoordinatorComment (type was based on "---_---_---_---_---_---_---_---_---_---" for shot 90091, could not convert string to float: ---
revert ECHCoordinatorComment to a string, type |S256
6 string reversions/compressions
['Status: |S1-> |S1', 'GasType: |S255-> |S8', 'CoordinatorComment: |S256-> |S171', 'LIDStatus: |S3-> |S3', 'ExperimentTheme: |S77-> |S77', 'ECHCoordinatorComment: |S255-> |S256']
"""

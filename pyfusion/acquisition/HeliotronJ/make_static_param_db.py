""" 
Extract the magnetic field and other static parameters of Heliotron J
into a dictionary of arrays

E4300: 90 entries/second excluding time in save_h_j_data


Uses exec of a dict - very fast!
 timeit exec('d=dict('+''.join(lines[5:-1]).replace(chr(0),'')+')') 144us
"""


import numpy as np


def params_to_dict(lines):
    """ read parameters from namelist output of Heliotron J static data
    into a dictionary(ies).  Returns a list of dictionaries.
    """
    lines = np.char.strip(lines)
    wp = np.where(np.char.find(lines,'DATAPARM')>-1)[0]
    we = np.where(np.char.rfind(np.char.add(lines,'@#@#'),'/@#@#') > -1)[0]

    if len(wp) != len(we):
        raise LookupError('different number of starts {s} and ends {s}'
                          .format(s=len(wp), e=len(we)))
    dics = []
    for (s,e) in zip(wp,we):
        dics.append(eval('dict('+''.join(lines[s+1:e]).replace(chr(0),'')+')'))
    return(dics)

def test(datafile='pyfusion/acquisition/HeliotronJ/params.out'):
    """
    >>> dics = test()
    >>> dics[0]['IBTA']
    -92
    """
    fp=open(datafile,'r')
    lines=fp.readlines()
    return(params_to_dict(lines))


import subprocess, os

def get_static_parms(shot, signal='DIA135',exe='save_h_j_data', press_on=True):

    cmd = str('echo {signal} {shot} nofile | {exe}'
              .format(signal=signal, shot=shot, exe=exe))
    
    parm_pipe = subprocess.Popen(cmd,  shell=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
    (resp,err) = parm_pipe.communicate()
    if (err != '') or (parm_pipe.returncode != 0): 
        if press_on: 
            print('ignoring failure to get params')
            return({})
        raise Exception('failed to run \n{cmd}\n{err}'.format(cmd=cmd, err=err))
    else:
        lines = resp
        if len(np.shape(lines))==0:
            for line_sep in ['\n\r','\r\n','\n','\r']:
                if line_sep in lines:
                    break
            lines = lines.split(line_sep)
        return(params_to_dict(lines))

def make_param_db(max_shot=70000, file_name='HJparams.npz', shot_range=range(50000,50010), signal='DIA135',db=None,exe='save_h_j_data'):
    if db is None:
        db = dict()
        for param in 'ISHOTNO,IBHV,IBTA,IBTB,IBAV,IBIV,ICH'.split(','):
            db.update({param: np.zeros(max_shot+1,dtype=np.int32)})

        db.update(dict(PANELDT = np.array((max_shot+1)*[100*' '])))
        db.update(dict(STIME = np.array((max_shot+1)*[5*' '])))
        db.update(dict(SDATE = np.array((max_shot+1)*[8*' '])))
    else:
        if len(db['ISHOTNO'])<max_shot:
            raise LookupError('database given is shorter ({dbl})'
                              ' than max_shot={m}'
                              .format(m=max_shot,dbl=len(db['ISHOTNO'])))


    this_file = os.path.abspath( __file__ )
    this_dir = os.path.split(this_file)[0]

    if not os.sep in exe :  # default to same path as this file
        exe = this_dir + os.sep + exe

    if not os.path.exists(exe):
        print('Can''t find exe \n{exe}\ndummy data using dummy exe "dummy_exe"'.
              format(exe=exe))
        exe = this_dir + os.sep + "dummy_exe"
        
    errors = []
    for shot in shot_range:
        try:
            params = get_static_parms(shot, signal=signal, exe=exe, press_on=False)
            for key in db.keys():
                db[key][shot] = params[0][key]
        except exception, reason:
            errors.append([shot, reason])

    args=','.join(["{k}=db['{k}']"
                   .format(k=k) for k in db.keys()])
    exec("np.savez_compressed(file_name,"+args+")")
    return(db, errors)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    dics = test()
    print("dics[0]['IBTA']) = {v}".format(v=dics[0]['IBTA']))
    

import os, sys
import pyfusion
""" Find binaries and compile if required

   The binaries are different for python2 and 3 etc, so name gethjdata accordingly
    We need to import different modules as gethjdata . To facilitata this, we need a 
   text-based import command that allows an 'as' cluase.
   This module exports either the standard import_module, or a home-made one, if the
   standard one will not do what we want, which seems to be the case

"""

# this is the right way, but I can't load mod as name this way
#    from importlib import import_module
if pyfusion.DEBUG>0: 
    print("Can't load via official import_module, trying a workaround using exec()")

def import_module(modstr, alt_name=None, dict1=None):
    if alt_name is None: alt_name = modstr
    if dict1 is None:
        raise Exception('need a dictionary in dict1 (usually locals())')
    else:
        exec('import {m} as {a}'.format(m=modstr, a=alt_name),globals(),dict1)

        


"""
#works
exec('import pyfusion.acquisition.HeliotronJ.gethjdata2_7 as gethjdata')
import_module('pyfusion.acquisition.HeliotronJ.gethjdata2_7')
import_module('.gethjdata2_7','pyfusion.acquisition.HeliotronJ')

#doesn't
import_module('gethjdata2_7','pyfusion.acquisition.HeliotronJ.gethjdata2_7')
"""

def get_hj_modules():
    # append python version number - but hide '.' 
    hj_module  = 'gethjdata'+sys.version[0:3].replace('.','_')
    # Note: 3.4 may need f2py3.4 - 3.5 f2py3 gives PyCObject_Type error
    f2py = 'f2py3' if sys.version >= '3,0' else 'f2py'
    oldwd = os.getcwd()

    short_host_name = os.uname()[1]
    cdir = os.path.dirname(os.path.abspath(__file__))
    #    cd = 'cd {cdir}; '.format(cdir=cdir)
    exe_path = os.path.join(cdir,short_host_name)
    if not(os.path.isdir(exe_path)):
        print('creating {d}'.format(d=exe_path))
        os.mkdir(exe_path)

    try:
        print('try import')
        import_module(hj_module,dict1=locals())
    except Exception as reason:
        print("Can't import {m} as get_hjdata at first attempt:  reason - {r}, {args}"
              .format(r=reason, args=reason.args, m=hj_module))
    # Should use subprocess instead of command, and get more feedback

        os.chdir(exe_path) # move to the exe dir (although module stays one up ../
        import subprocess
        print('Compiling Heliotron J data aquisition library, please wait...')
        ## Note: g77 will do, (remove --fcompiler-g95)  but can't use TRIM function etc 
        cmds = ['gcc -c -fPIC ../libfdata.c ../intel.c',
                # f2py won't accept ../ or full path in the -m param! need to cd
                'cd ..; {f} --fcompiler=gfortran -c -m {m} {xp}.o -lm  hj_get_data.f'
                .format(m=hj_module, f=f2py,xp=os.path.join(exe_path,'*')),
                'f77 -Lcdata ../save_h_j_data.f intel.o libfdata.o -o {exe}'
                .format(exe=os.path.join(exe_path,'save_h_j_data')), # 2015
            ]
        for cmd in cmds:
            sub_pipe = subprocess.Popen(cmd,  shell=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            (resp,err) = sub_pipe.communicate()
            if (err != b'') or (sub_pipe.returncode != 0): 
                print(resp,err,'.') #
                print(err.split(b'\n')[-2]),
            if resp !='': print(resp[-10:])

        try:
            print('try after compiling...'),
            import_module(hj_module)
        except Exception as reason:
            print("Can't import {m} as get_hjdata at second attempt {r}, {args}"
                  .format(r=reason, args=reason.args, m=hjmod))
            raise ImportError("Can't import Heliotron J data acquisition library")
    finally:
        os.chdir(oldwd)
        return(hj_module, exe_path)

    # Dave had some reason for not including the auto compile - Boyd added 2013
    # probably should suppress the auto compile during tests - this was his code.
    #except:
    #    # don't raise an exception - otherwise tests will fail.
    #    # TODO: this should go into logfile
    #    print ImportError, "Can't import Heliotron J data aquisition library"


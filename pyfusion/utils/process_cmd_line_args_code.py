_thisdoc=\
"""
Uses exec now - python 3 compatible (not execfile)
version 2013 - includes _var_defaults (which has comments as well as defaults)

This is code to be "inlined" to allow python statments to be passed
 from the command line.  New version ~327 checks for the existence 
of the LHS of an assignmment, and adds quotes if that target is a string.  
Using None - 
Note that nested quotes may still be needed if string or RHS contains strange chars,
or if the current value of the target is NOT a string (e.g. None)  
  run pyfusion/examples/clean_up_pyfusion_text.py "fileglob='PF2_130812_MP201*2'" 
None will also allow evaluation of functions (whereas a string initial value won't)
  run pyfusion/examples/clean_up_pyfusion_text.py fileglob=glob.glob('PF2_130812_MP201*2')[0] 

This version will work inside, or independent of, pyfusion, and will follow 
a local variable <verbose>, or generate its own.
Usage:
 execfile("process_cmd_line_args.py")
Note that by putting thisdoc= at the top, this doens't overwrite the 
caller's __doc__ string, and muck up help etc

Note: if your mistype _var_defaults, it will work until there is an error reading the args
for example, many routines use _var_default mistakenly

Note that the ipython run command simulates a "fresh " run, unless -i 
is used,  so the locals and globals in ipython will not available for use. 
One downside of using -i is the local dictionary is likely to be large,
so debugging a problem may be more difficult.  

See argparse for some new argument features - deals with
similar functions, includes options, but does not replace
process_cmd_line_args whose specialty is variable assignments.

Problem - process_cmd_line_args.py needs to be in the current directory.
This can be solved by rewriting as  (both of which are valid python 3)
  a/ exec(pyfusion.utils.process_cmd_line_args)         
  b/ pyfusion.utils.process_cmd_line_args(locals())  ** preferred solution
In a), the function process_cmd_line_args finds the file process_cmd_line_args.py in the PYTHONPATH,
      so the function process_cmd_line_args is completley different to the file of that name.
In b), the local dictionary is updated.  b) might be clearer to do it explicitly
     locals().update(pyfusion.utils.process_cmd_line_args(locals()))
"""
# cant deepcopy locals()
import sys as _sys
import os
import six
from six.moves import input

## put any functions you might use in the expressions here
from numpy import arange, array, sort
try:
    verbose=pyfusion.settings.VERBOSE
    from pyfusion.utils import get_local_shot_numbers
except:
    if 'verbose' not in locals() and 'verbose' not in globals():
        verbose=2
        print(' process_cmd_line_args detected we are running outside of'
              ' pyfusion, verbose=%d' % verbose)

# from matplotlib import is_string_like # deprecated in 2.2

def list_vars(locdict, Stop, tail_msg=''):
    if '_var_defaults' in locdict:
        print('\n===== Input variables, and their default values '
              '(from _var_defaults) ======')
        print(locdict['_var_defaults'])
    # check the global namespace too - can't see _var_defaults when
    # running with "run -i" (but it hasn't helped).    
    if '_var_defaults' in globals():
        print('\n=========== Variables, and default values =========')
        print(globals()['_var_defaults'])
    else:

        _user_locals=[]
        for v in locdict:
            if (v.find('_')!=0
                 and str(locdict[v]).find('function')<0
                 and str(locdict[v]).find('module')<0): 
                _user_locals.append(v)
        print('\n========= Accessible variables and current values are: =====')  #, _user_locals)
        if verbose > 0:
            print('locals............')
            _n=0
            for k in _user_locals:
                print("  %s = %s" %  (k, locdict[k]))
                _n +=1
                if (_n==20):
                    # was raw_input for python2 - six.moves works for python2/3
                    ans=input('do you want to see the rest of the local vars? (y/N/Q) ')
                    if  ans.upper()=='Q': 
                        _sys.exit()

                    elif ans.upper()!='Y': 
                        print(ans)
                        break

        if _rhs != None and _rhs != "None":
            if not _rhs in locdict:
                print('RHS < {rhs} > is not in local dictionary - if you wish to refer to '
                       'a variable from the working interactive namespace, then '
                       'use the -i option (under ipython only)'.format(rhs=_rhs))
        # Note: pydb is nicer but slower....                     
    if Stop: 
        print('======== make sure there are *NO SPACES* - e.g.  x=123  not x = 123 ======')
        if tail_msg !='': print(tail_msg)
        ans=input(' q or ^C (+<CR>) to stop? ')  # raw_input still needs a CR 
        if ans.upper() == 'Q': 
            _sys.exit()
        return()  # I thought I didn't know how to just "stop" - maybe the above works
    try:
        import pydb; pydb.set_trace('s','print "set vars, c to continue"')
    except:
        print('unable to load pydb, using pdb')
        import pdb; pdb.set_trace()
    'c to continue, or s for one step, then enter the var name manually '


# "main" code  (although not really - just the code executed inline
# override the defaults that have been set before execfile'ing' this code
# exec is "built-in" apparently
   
_loc_dict=locals().copy() # need to save, as the size changes
if verbose>1: print ('{n} args found'.format(n=len(_sys.argv)))
if verbose>1: print(' '.join([_arg for _arg in _sys.argv]))
_rhs=None
# this would be a way to ignore spaces around equals, but need to preserve 
# spaces between statements! leave for now
#_expr=string.join(_sys.argv[1:])
# check if the argv is what we expect - (needed to work within emacs - argv[0] is ipython
_args = _sys.argv[:]
if os.path.split(_args[0])[-1] in ['ipython']:
    _args = []  # why wipe them out?
else:     # argv[0] is hopefully a python script, and we don't want to parse it
    _args = _args[1:]

print('Using pyfusion.utils.process_cmd_line_args_code...')
if 'from_runpy' in globals():  # Note: this refers to runpy_cmd not the usual runpy
    _args = args_from_runpy.split(' ')
for _expr in _args:
    if (array(_expr.upper().split('-')) == "HELP").any():
        if '__doc__' in _loc_dict:
            print("\n======Documentation from caller's source file "
                  "(__doc__ captures the first comment ===\n")
            print(_loc_dict['__doc__'])
        else: print('No local help')
        list_vars(_loc_dict, Stop=True)
    else:
        if _expr.startswith('--'):
            print('skipping ' + _expr)
            continue
        if verbose>3: print('assigning %s from command line') % _expr
        _words=_expr.split('=')
        _firstw=_words[0]
        _lhs=_firstw.strip()  # remove excess spaces
        if len(_words)>1:_lastw=_words[1]
        else: _lastw = ""
        _rhs=_lastw.strip()
        try:
            _expr_to_exec = _lhs
            exec(_expr_to_exec)  # this tests for existence of the LHS (target)
            # if the present _lhs contains a string, try adding quotes to RHS
            # in case they were stripped by the shell.  But won't work for 
            # _lhs that are None.
            _expr_to_exec = 'is_str = isinstance('+_lhs+', six.string_types)'
            exec(_expr_to_exec)
            if is_str and _rhs[0]!="'" and _rhs[0]!='"':
                _expr_to_exec = _lhs+'="'+_rhs+'"'
            else: _expr_to_exec = _expr  # the original expr
            if verbose>3: print('actual statement: {ee}'
                                .format(ee=_expr_to_exec))
            exec(_expr_to_exec)
        except Exception as _info: # _info catches the exception info
            err=str("######Target variable {lh} not set or non-existent!#####"
                  "\n executing {ex}, original rhs was {rh} (quotes gobbled?)\n"
                  .format(lh=_lhs, ex=_expr_to_exec, rh=_rhs))
            err2=str('< %s > raised the exception < %s >' % (_expr,_info))
            _sys.stderr.write(err)
            _sys.stderr.write(err2)
# list_vars will also offer to enter a debugger..
            list_vars(_loc_dict, Stop=True, tail_msg=err2)

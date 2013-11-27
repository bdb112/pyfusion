_thisdoc=\
"""
* Note: this is probably the last version of the execfile version: see below.

version 2013 - includes _var_defaults (which has comments as well as defaults)
This is code to be "inlined" to allow python statments to be passed
 from the command line.  New version ~327 checks for the existence 
of the LHS of an assignmment, and adds quotes if that target is a string.  
Note that quotes may still be needed if string or RHS contains strange chars,
or if the current value of the target is NOT a string (e.g. None)

This version will work inside, or independent of, pyfusion, and will follow 
a local variable <verbose>, or generate its own.
Usage:
 execfile("process_cmd_line_args.py")
Note that by putting thisdoc= at the top, this doens't overwrite the 
caller's __doc__ string, and muck up help etc

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
## put any functions you might use in the expressions here
from numpy import arange, array, sort
try:
    verbose=pyfusion.settings.VERBOSE
    from pyfusion.utils import get_local_shot_numbers
except:
    if not(locals().has_key('verbose')):
        verbose=2
        print(' process_cmd_line_args detected we are running outside of'
              ' pyfusion, verbose=%d' % verbose)
import string
from pylab import is_string_like

def list_vars(locdict, Stop):
    if locdict.has_key('_var_defaults'):
        print('\n=========== Variables, and default values =========')
        print(locdict['_var_defaults'])
    # check the global namespace too - can't see _var_default when
    # running with "run -i" (but it hasn't helped).    
    if globals().has_key('_var_defaults'):
        print('\n=========== Variables, and default values =========')
        print(globals()['_var_defaults'])
    else:

        _user_locals=[]
        for v in locdict:
            if (v.find('_')!=0
                 and str(locdict[v]).find('function')<0
                 and str(locdict[v]).find('module')<0): 
                _user_locals.append(v)
        print '\n========= Accessible variables and current values are: ====='  #, _user_locals
        if verbose>0:
            _n=0
            for k in _user_locals:
                print("  %s = %s" %  (k, locdict[k]))
                _n +=1
                if (_n==20):
                    ans=raw_input('do you want to see the rest of the local vars? (y/N) ')
                    if ans.upper()!='Y': 
                        print(ans)
                        break
        if _rhs != None:
            if not(locdict.has_key(_rhs)): 
                print(('RHS < %s > is not in local dictionary - if you wish to refer to '
                       'a variable from the working interactive namespace, then '
                       'use the -i option (under ipython only)') % _rhs)
        # Note: pydb is nicer but slower....                     
    if Stop: 
        print('======== make sure there are no spaces - e.g.  x=123  not x = 123 ======')
        ans=raw_input(' ^C to stop')
        return()  # don't know how to just "stop"
    try:
        import pydb; pydb.set_trace('s','print "set vars, c to continue"')
    except:
        print('unable to load pydb, using pdb')
        import pdb; pdb.set_trace()
    'c to continue, or s for one step, then enter the var name manually '


# "main" code  (although not really - just the code executed inline
# override the defaults that have been set before execfile'ing' this code
# exec is "built-in" apparently
   
if verbose>1: print ('%d args found') % (len(_sys.argv))
if verbose>1: print(' '.join([_arg for _arg in _sys.argv]))
_rhs=None
# this would be a way to ignore spaces around equals, but need to preserve 
# spaces between statements! leave for now
#_expr=string.join(_sys.argv[1:])
for _expr in _sys.argv[1:]:
    if (array(_expr.upper().split('-')) == "HELP").any():
        if locals().has_key('__doc__'): 
            print(" ==================== printing local __doc__ (from caller's source file) ===")
            print locals()['__doc__']
        else: print('No local help')
        list_vars(locals(), Stop=True)
    else:
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
            _expr_to_exec = 'is_str = is_string_like('+_lhs+')'
            exec(_expr_to_exec)
            if is_str and _rhs[0]!="'" and _rhs[0]!='"':
                _expr_to_exec = _lhs+'="'+_rhs+'"'
            else: _expr_to_exec = _expr
            if verbose>3: print('actual statement: %s') % _expr_to_exec
            exec(_expr_to_exec)
        except Exception, _info: # _info catches the exception info
            err=str("######Target variable {lh} not set or non-existent!#####"
                  "\n executing {ex}, original rhs was {rh}"
                  .format(lh=_lhs, ex=_expr_to_exec, rh=_rhs))
            err2=str('< %s > raised the exception < %s >' % (_expr,_info))
            _sys.stderr.write(err)
            _sys.stderr.write(err2)
            _loc_dict=locals().copy() # need to save, as the size changes
# list_vars will also offer to enter a debugger..
            list_vars(_loc_dict, Stop=True)


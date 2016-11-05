""" 
Extract LP information from the pyfusion cfg files and print in an orderly way.
Useful for cross-checking.

Can either have one option/line, and no irrelevant diags or all options
on one line, but then irrelevant diags are printed as well
"""

from __future__ import print_function
import pyfusion
import numpy as np      

option_list = 'coords_w7_x_koord,area,params,sweepv,gain'
#option_list = 'sweepv'
#option_list = 'params'
#option_list = 'gain'

allsects = pyfusion.config.sections()
allsort = np.sort(allsects)

allvals = []
dups = []
one_col = False
#allsort = ['Diagnostic:W7X_L53_LP01_I']

allprobes = []
for sec in allsort:
    toks = sec.split(':')
    if toks[0].strip() != 'Diagnostic':
        continue
    if 'W7X_L5' not in toks[1]:
        continue
    if not toks[1].endswith('_I'):
        continue

    allprobes.append(toks[1])

for diag in allprobes:
    #  | allows separating excel cells; If we use : -> confusion by dictionaries
    if not one_col:
        print(diag, end=' | ') 
    
    ops = pyfusion.config.pf_options('Diagnostic', diag)
    for op in option_list.split(','):
        if op in ops:
            if one_col: print(diag, end=' | ')  # this is for one item/line
            val = pyfusion.config.pf_get('Diagnostic', diag, op)
            #print('[{diag}]'.format(diag=diag))
            val = pyfusion.config.get('Diagnostic:'+diag, op)
            if '=' in val:  # digest dictionaries to rationalise representation
                val = eval('dict({v})'.format(v=val))
            elif '(' in val: # also tuples
                val = eval(val)
            elif ',' in val: # and lists without parens
                val = eval('({v})'.format(v=val))
            if val in allvals:
                dups.append([diag,op,val])
            allvals.append(val)
            end = ' \n ' if one_col else ' | '
            print('{op} = {val}'
                  .format(op=op, val=val), end=end)
    if not one_col: print()

if len(dups)>0:
    print(len(dups),'dups')
    
